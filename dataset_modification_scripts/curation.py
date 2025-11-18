from loguru import logger
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable, Tuple, Union, Type, Literal

import requests
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm

from src.utils.config import (
    ensure_dir,
    load_config_with_defaults,
)
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()


class RetryConfig:
    def __init__(self, max_retries: int = 3, backoff_seconds: float = 2.0):
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds


class FireworksChatClient:
    """Fireworks chat interface using the OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str,
        api_key_env: str = "FIREWORKS_API_KEY",
        base_url: str = "https://api.fireworks.ai/inference/v1",
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        retry: Optional[RetryConfig] = None,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing {api_key_env} in environment. Ensure it is set (e.g., via .env)."
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.stop = stop
        self.retry = retry or RetryConfig()
        self.response_model = response_model

    def _json_schema_from_model(self, model: Type[BaseModel]) -> Dict[str, Any]:
        return model.model_json_schema()  # type: ignore[no-any-return]

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Optional[Type[BaseModel]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        last_err: Optional[Exception] = None
        for attempt in range(self.retry.max_retries + 1):
            try:
                model_for_response = response_model or self.response_model
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self.top_p is not None:
                    kwargs["top_p"] = self.top_p
                if self.stop is not None:
                    kwargs["stop"] = self.stop
                if model_for_response is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": model_for_response.__name__,
                            "schema": self._json_schema_from_model(model_for_response),
                        },
                    }
                response = self.client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < self.retry.max_retries:
                    time.sleep(self.retry.backoff_seconds * (2**attempt))
                else:
                    break
        raise last_err or RuntimeError(
            "Fireworks generation failed for unknown reasons"
        )


class FireworksBatchClient:
    """Fireworks batch REST client for dataset uploads and batch inference."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        api_key_env: str = "FIREWORKS_API_KEY",
        account_id: str = "geodesic-puria",
        batch_base_url: str = "https://api.fireworks.ai/v1",
        http_timeout: float = 60.0,
        temperature: float = 0.2,
        max_tokens: int = 256,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> None:
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Missing {api_key_env} in environment. Ensure it is set (e.g., via .env)."
            )
        self.api_key = api_key
        self.model = model
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.top_p = top_p
        self.stop = stop
        self.account_id = account_id
        self.batch_base_url = batch_base_url.rstrip("/")
        self.http_timeout = float(http_timeout)

    @staticmethod
    def write_batch_input_file(
        records: Iterable[Dict[str, Any]],
        output_path: Union[str, Path],
        ensure_ascii: bool = False,
    ) -> Path:
        """Serialize batch records to JSONL while validating required fields.

        If a record is missing 'custom_id', it will be automatically generated
        as 'request-{index}' where index is the 0-based position in the sequence.
        The 'body' field is still required.
        """
        path = Path(output_path)
        ensure_dir(str(path.parent))
        count = 0
        with path.open("w", encoding="utf-8") as fp:
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError(
                        "Each record must be a dict with at least a 'body' field."
                    )
                if "body" not in record:
                    raise ValueError("Batch records must include a 'body' field.")
                # Auto-generate custom_id if missing
                if "custom_id" not in record:
                    record = record.copy()  # Avoid mutating the original
                    record["custom_id"] = f"request-{count}"
                fp.write(json.dumps(record, ensure_ascii=ensure_ascii))
                fp.write("\n")
                count += 1
        if count == 0:
            raise ValueError("Cannot create an empty batch dataset.")
        return path

    def _require_account_id(self) -> str:
        return self.account_id

    def _batch_headers(
        self, content_type: Optional[str] = "application/json"
    ) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

    def _request_batch_api(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_payload: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Any:
        url = f"{self.batch_base_url}{path}"
        merged_headers = self._batch_headers() if headers is None else headers
        response = requests.request(
            method,
            url,
            params=params,
            json=json_payload,
            data=data,
            files=files,
            headers=merged_headers,
            timeout=self.http_timeout,
        )
        if not response.ok:
            error_msg = f"HTTP {response.status_code} {response.reason}"
            error_details = None
            try:
                error_body = response.json()
                error_details = error_body
                if isinstance(error_body, dict):
                    # Try various common error message fields
                    if "error" in error_body:
                        if isinstance(error_body["error"], dict):
                            error_msg += f": {json.dumps(error_body['error'])}"
                        else:
                            error_msg += f": {error_body['error']}"
                    elif "message" in error_body:
                        error_msg += f": {error_body['message']}"
                    elif "detail" in error_body:
                        error_msg += f": {error_body['detail']}"
                    else:
                        error_msg += f": {json.dumps(error_body)}"
                else:
                    error_msg += f": {response.text[:500]}"
            except (ValueError, KeyError):
                error_msg += f": {response.text[:500]}"

            logger.error(f"Batch API request failed: {error_msg}")
            logger.error(f"Request URL: {url}")
            logger.error(
                f"Request payload: {json.dumps(json_payload, indent=2) if json_payload else 'None'}"
            )
            if error_details:
                logger.error(f"Error response: {json.dumps(error_details, indent=2)}")

            # Raise with more context
            raise requests.exceptions.HTTPError(
                f"{error_msg}\nURL: {url}\nPayload: {json.dumps(json_payload, indent=2) if json_payload else 'None'}",
                response=response,
            )
        response.raise_for_status()
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError:
            return response.content

    def _qualify_dataset_id(self, dataset_id: str) -> str:
        if dataset_id.startswith("accounts/"):
            return dataset_id
        account_id = self._require_account_id()
        return f"accounts/{account_id}/datasets/{dataset_id}"

    def _dataset_resource_path(self, dataset_id: str) -> str:
        """Return the dataset identifier portion expected in REST URL paths."""
        if dataset_id.startswith("accounts/"):
            parts = dataset_id.split("/datasets/", 1)
            if len(parts) == 2 and parts[1]:
                return parts[1]
        return dataset_id

    def create_batch_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """Create a Fireworks dataset placeholder for batch uploads."""
        account_id = self._require_account_id()
        payload = {"datasetId": dataset_id, "dataset": {"userUploaded": {}}}
        return self._request_batch_api(
            "POST",
            f"/accounts/{account_id}/datasets",
            json_payload=payload,
        )

    def upload_batch_dataset_file(
        self, dataset_id: str, file_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Upload a JSONL file to a Fireworks dataset."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        account_id = self._require_account_id()
        target = self._dataset_resource_path(dataset_id)
        url_path = f"/accounts/{account_id}/datasets/{target}:upload"
        with path.open("rb") as fp:
            files = {"file": (path.name, fp, "application/jsonl")}
            return self._request_batch_api(
                "POST",
                url_path,
                files=files,
                headers=self._batch_headers(content_type=None),
            )

    def list_batch_datasets(self, page_token: Optional[str] = None) -> Dict[str, Any]:
        """List all batch datasets for the configured account.

        Returns:
            Dict with 'datasets' list and optional 'nextPageToken' and 'totalSize'
        """
        account_id = self._require_account_id()
        params = {}
        if page_token:
            params["pageToken"] = page_token
        return self._request_batch_api(
            "GET",
            f"/accounts/{account_id}/datasets",
            params=params if params else None,
        )

    def get_batch_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific dataset by ID, or None if it doesn't exist.

        This method searches through listed datasets to find a match.
        Uses the list endpoint to avoid 404 error logs when checking existence.
        """
        dataset_path = self._dataset_resource_path(dataset_id)

        # Use list endpoint to avoid 404 error logs when dataset doesn't exist
        # Search through listed datasets
        page_token = None
        dataset_id = self._qualify_dataset_id(dataset_id)
        while True:
            response = self.list_batch_datasets(page_token=page_token)
            datasets = response.get("datasets", [])

            # Search for matching dataset by name (dataset_id)
            for dataset in datasets:
                # Match by name field (which should be the dataset_id)
                # Also check if the qualified path matches
                dataset_name = dataset.get("name", "")
                if dataset_name == dataset_id or dataset_name == dataset_path:
                    return dataset

            # Check if there are more pages
            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return None

    def dataset_exists(self, dataset_id: str) -> bool:
        """Check if a dataset exists by attempting to retrieve it."""
        return self.get_batch_dataset(dataset_id) is not None

    def delete_batch_dataset(
        self, dataset_id: str, check_exists: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Delete a Fireworks dataset.

        Args:
            dataset_id: The dataset ID to delete
            check_exists: If True, check if dataset exists before attempting deletion

        Returns:
            Dict response from API, or None if dataset doesn't exist and check_exists=True

        Raises:
            requests.exceptions.HTTPError: If deletion fails (unless 404 and check_exists=False)
        """
        if check_exists:
            if not self.dataset_exists(dataset_id):
                logger.debug(f"Dataset {dataset_id} does not exist, skipping delete")
                return None

        account_id = self._require_account_id()
        dataset_path = self._dataset_resource_path(dataset_id)
        try:
            return self._request_batch_api(
                "DELETE",
                f"/accounts/{account_id}/datasets/{dataset_path}",
            )
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.debug(f"Dataset {dataset_id} does not exist (404)")
                return None
            raise

    def create_batch_job(
        self,
        job_id: str,
        *,
        input_dataset_id: str,
        output_dataset_id: str,
        model: Optional[str] = None,
        inference_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Launch a Fireworks batch inference job."""
        account_id = self._require_account_id()
        model_name = model or self.model
        if not model_name:
            raise ValueError(
                "Model must be provided either during client initialization or when creating the batch job."
            )
        payload = {
            "model": model_name,
            "inputDatasetId": self._qualify_dataset_id(input_dataset_id),
            "outputDatasetId": self._qualify_dataset_id(output_dataset_id),
        }
        if inference_parameters is not None:
            payload["inferenceParameters"] = inference_parameters
        else:
            params = {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            }
            if self.top_p is not None:
                params["topP"] = self.top_p
            if self.stop is not None:
                params["stopSequences"] = self.stop
            payload["inferenceParameters"] = params
        return self._request_batch_api(
            "POST",
            f"/accounts/{account_id}/batchInferenceJobs",
            params={"batchInferenceJobId": job_id},
            json_payload=payload,
        )

    def get_batch_job(self, job_id: str) -> Dict[str, Any]:
        """Fetch metadata for a specific batch inference job."""
        account_id = self._require_account_id()
        return self._request_batch_api(
            "GET",
            f"/accounts/{account_id}/batchInferenceJobs/{job_id}",
        )

    def list_batch_jobs(self) -> Dict[str, Any]:
        """Return all batch inference jobs for the configured account."""
        account_id = self._require_account_id()
        return self._request_batch_api(
            "GET",
            f"/accounts/{account_id}/batchInferenceJobs",
        )

    def get_dataset_download_urls(self, dataset_id: str) -> Dict[str, Any]:
        """Retrieve signed URLs for every object in an output dataset."""
        account_id = self._require_account_id()
        dataset_path = self._dataset_resource_path(dataset_id)
        return self._request_batch_api(
            "POST",
            f"/accounts/{account_id}/datasets/{dataset_path}:getDownloadEndpoint",
            json_payload={},
        )

    def download_batch_outputs(
        self,
        dataset_id: str,
        output_dir: Union[str, Path],
    ) -> List[Path]:
        """Download all files referenced by a dataset download manifest."""
        manifest = self.get_dataset_download_urls(dataset_id)
        filename_map = manifest.get("filenameToSignedUrls", {})
        if not filename_map:
            return []
        target_dir = Path(output_dir)
        ensure_dir(str(target_dir))
        downloaded: List[Path] = []
        for object_path, signed_url in filename_map.items():
            fname = Path(object_path).name
            destination = target_dir / fname
            response = requests.get(signed_url, timeout=self.http_timeout)
            response.raise_for_status()
            destination.write_bytes(response.content)
            downloaded.append(destination)
        return downloaded


def format_prompt(template: str, example: Dict[str, Any]) -> str:
    """Render a simple string template using example keys via {key} placeholders."""
    try:
        return template.format(**example)
    except KeyError:
        # Fallback: provide the raw example as JSON when keys mismatch
        return f"Example JSON for reference:\n```json\n{json.dumps(example, ensure_ascii=False, indent=2)}\n```\n\nTemplate:\n{template}"


class JudgeVerdictModel(BaseModel):
    reason: str
    verdict: Literal["HIGH", "LOW"]


class DatasetSampleModel(BaseModel):
    """Pydantic model for dataset sample format."""

    prompt_list: List[str]
    high_reward_answer: str
    other_answers: List[str]
    correct_answer: str


class QualityJudge:
    """LLM-based judge that classifies examples as high/low quality from a parametric prompt.

    Supports two LLM output formats:
    1. JSON: {"verdict":"HIGH","reason":"..."}
    2. Plain text containing the tokens HIGH or LOW
    """

    def __init__(
        self,
        llm: FireworksChatClient,
        system_prompt: str,
        user_prompt_template: str,
        high_token: str = "HIGH",
        low_token: str = "LOW",
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.high_token = high_token
        self.low_token = low_token

    def is_high_quality(
        self, example: Dict[str, Any]
    ) -> Tuple[bool, str, Optional[str]]:
        """Returns (is_high_quality, verdict_text, reason)."""
        if example["correct_answer"] == example["high_reward_answer"]:
            # Save money!
            return (
                False,
                "LOW",
                "AUTO: The correct answer is the same as the high-reward answer.",
            )
        prompt = format_prompt(self.user_prompt_template, example)
        verdict_text = self.llm.chat(self.system_prompt, prompt)
        try:
            parsed = json.loads(verdict_text)
            if isinstance(parsed, dict) and "verdict" in parsed:
                verdict_label = str(parsed.get("verdict", "")).upper()
                reason = parsed.get("reason", "")
                if verdict_label in [self.high_token.upper(), self.low_token.upper()]:
                    return (
                        verdict_label == self.high_token.upper(),
                        verdict_label,
                        reason,
                    )

                else:
                    raise ValueError("Invalid verdict label in JSON.")
            else:
                raise ValueError("JSON not in expected dict format.")
        except (json.JSONDecodeError, ValueError, TypeError):
            raise ValueError("Invalid verdict JSON.")


class DatasetCurator:
    """Filter a dataset using an LLM judge and persist high-quality examples."""

    def __init__(
        self,
        judge: Optional[QualityJudge] = None,
        keep_fields: Optional[List[str]] = None,
    ) -> None:
        self.judge = judge
        self.keep_fields = keep_fields

    def _select_fields(self, example: Dict[str, Any]) -> Dict[str, Any]:
        if not self.keep_fields:
            return example
        return {k: example.get(k) for k in self.keep_fields}

    def filter_dataset(
        self,
        dataset: Union[DatasetDict, Dataset],
        max_samples: Optional[int] = None,
        progress: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return (kept_examples, rejected_examples_with_meta)."""
        if self.judge is None:
            raise ValueError("judge must be provided to filter_dataset")

        def iter_examples(ds: Union[DatasetDict, Dataset]) -> Iterable[Dict[str, Any]]:
            if isinstance(ds, DatasetDict):
                # Prefer 'train' split if present
                if "train" in ds:
                    yield from ds["train"]
                else:
                    for split in ds.keys():
                        yield from ds[split]
            else:
                yield from ds

        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        iterator: Iterable[Dict[str, Any]] = iter_examples(dataset)
        iterator = tqdm(iterator, disable=not progress, desc="Judging examples")

        for idx, ex in enumerate(iterator):
            if max_samples is not None and idx >= max_samples:
                break
            is_high, verdict, reason = self.judge.is_high_quality(ex)
            if is_high:
                logger.info(f"LLM accepted sample {idx}: {reason}")
                kept.append(self._select_fields(ex))
            else:
                logger.warning(f"LLM rejected sample {idx}: {reason}")
                rejected.append({"example": ex, "verdict": verdict})

        return kept, rejected

    def save_jsonl(self, examples: List[Dict[str, Any]], path: str) -> None:
        ensure_dir(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def filter_dataset_batch(
        self,
        dataset: Union[DatasetDict, Dataset],
        batch_client: FireworksBatchClient,
        system_prompt: str,
        user_prompt_template: str,
        max_samples: Optional[int] = None,
        batch_dataset_id: Optional[str] = None,
        batch_job_id: Optional[str] = None,
        temp_dir: Optional[str] = None,
        poll_interval: float = 10.0,
        max_wait_time: float = 36000.0,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter dataset using Fireworks batch API.

        Args:
            dataset: Dataset to filter
            batch_client: FireworksBatchClient instance
            system_prompt: System prompt for the judge
            user_prompt_template: Template for user prompts (with {key} placeholders)
            max_samples: Maximum number of samples to process
            batch_dataset_id: Optional dataset ID for batch input. If None, auto-generated.
            batch_job_id: Optional job ID for batch inference. If None, auto-generated.
            temp_dir: Temporary directory for batch files. If None, uses system temp.
            poll_interval: Seconds between polling batch job status
            max_wait_time: Maximum seconds to wait for batch job completion

        Returns:
            Tuple of (kept_examples, rejected_examples_with_meta)
        """
        import tempfile
        import uuid

        def iter_examples(ds: Union[DatasetDict, Dataset]) -> Iterable[Dict[str, Any]]:
            if isinstance(ds, DatasetDict):
                if "train" in ds:
                    yield from ds["train"]
                else:
                    for split in ds.keys():
                        yield from ds[split]
            else:
                yield from ds

        # Collect examples
        examples = []
        iterator = iter_examples(dataset)
        for idx, ex in enumerate(iterator):
            if max_samples is not None and idx >= max_samples:
                break
            examples.append((idx, ex))

        if len(examples) == 0:
            return [], []

        # Prepare batch input records
        logger.info(f"Preparing {len(examples)} examples for batch processing")
        batch_records = []
        for idx, ex in examples:
            # Format user prompt
            user_prompt = format_prompt(user_prompt_template, ex)

            # Check auto-reject condition
            if ex.get("correct_answer") == ex.get("high_reward_answer"):
                # Skip batch API call for auto-rejected examples
                continue

            # Create batch request body
            # Note: model and inference parameters are specified at the job level, not in individual requests
            # However, response_format may need to be in each request body
            body = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "JudgeVerdictModel",
                        "schema": JudgeVerdictModel.model_json_schema(),
                    },
                },
            }

            batch_records.append(
                {
                    "custom_id": f"example-{idx}",
                    "body": body,
                }
            )

        if len(batch_records) == 0:
            logger.warning("No examples to process via batch API (all auto-rejected)")
            # Handle auto-rejected examples
            rejected = []
            for idx, ex in examples:
                if ex.get("correct_answer") == ex.get("high_reward_answer"):
                    rejected.append(
                        {
                            "example": ex,
                            "verdict": "LOW",
                            "reason": "AUTO: The correct answer is the same as the high-reward answer.",
                        }
                    )
            return [], rejected

        # Setup temporary directory
        if temp_dir is None:
            temp_dir_obj = tempfile.mkdtemp(prefix="curation_batch_")
            temp_dir = temp_dir_obj
        else:
            ensure_dir(temp_dir)
            temp_dir_obj = None

        try:
            # Generate IDs if not provided
            # Use timestamp + UUID for output dataset to ensure uniqueness (Fireworks doesn't allow reusing output dataset IDs)
            if batch_dataset_id is None:
                batch_dataset_id = f"curation-input-{uuid.uuid4().hex[:8]}"
            if batch_job_id is None:
                batch_job_id = f"curation-job-{uuid.uuid4().hex[:8]}"
            # Include timestamp to ensure output dataset ID is always unique
            # Fireworks requires IDs to only contain lowercase a-z, 0-9, and hyphens
            timestamp = int(time.time() * 1000)  # milliseconds
            output_dataset_id = f"curation-output-{timestamp}-{uuid.uuid4().hex[:8]}"

            # Write batch input file
            input_file = Path(temp_dir) / "batch_input.jsonl"
            FireworksBatchClient.write_batch_input_file(batch_records, input_file)
            logger.info(f"Wrote batch input file: {input_file}")

            # Delete existing datasets if they exist (Fireworks doesn't allow reusing existing datasets)
            # Check and delete input dataset
            if batch_client.dataset_exists(batch_dataset_id):
                logger.info(f"Input dataset {batch_dataset_id} exists, deleting...")
                result = batch_client.delete_batch_dataset(batch_dataset_id)
                if result is not None:
                    logger.info(
                        f"Successfully deleted input dataset: {batch_dataset_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to delete input dataset: {batch_dataset_id}"
                    )
            else:
                logger.debug(
                    f"Input dataset {batch_dataset_id} does not exist, skipping delete"
                )

            # Check and delete output dataset (Fireworks will create it automatically during batch job)
            if batch_client.dataset_exists(output_dataset_id):
                logger.info(f"Output dataset {output_dataset_id} exists, deleting...")
                result = batch_client.delete_batch_dataset(output_dataset_id)
                if result is not None:
                    logger.info(
                        f"Successfully deleted output dataset: {output_dataset_id}"
                    )
                else:
                    logger.warning(
                        f"Failed to delete output dataset: {output_dataset_id}"
                    )
            else:
                logger.debug(
                    f"Output dataset {output_dataset_id} does not exist (expected)"
                )

            # Create and upload input dataset
            logger.info(f"Creating batch dataset: {batch_dataset_id}")
            batch_client.create_batch_dataset(batch_dataset_id)
            batch_client.upload_batch_dataset_file(batch_dataset_id, input_file)
            logger.info(f"Uploaded batch input dataset: {batch_dataset_id}")

            # Note: We don't create the output dataset here - Fireworks will create it
            # automatically when the batch job runs. This avoids the "already created" error.
            logger.info(
                f"Output dataset {output_dataset_id} will be created by Fireworks during batch job"
            )

            # Create batch job with inference parameters
            # Note: response_format is included in individual request bodies, not here
            logger.info(f"Creating batch job: {batch_job_id}")
            batch_client.create_batch_job(
                batch_job_id,
                input_dataset_id=batch_dataset_id,
                output_dataset_id=output_dataset_id,
            )
            logger.info(f"Batch job created. Waiting for completion...")

            # Poll for job completion
            start_time = time.time()
            while True:
                job_status = batch_client.get_batch_job(batch_job_id)
                status = job_status.get("status", "UNKNOWN")
                logger.info(f"Batch job status: {status}")

                if status in ["COMPLETED", "FAILED", "CANCELLED"]:
                    break

                elapsed = time.time() - start_time
                if elapsed > max_wait_time:
                    raise TimeoutError(
                        f"Batch job {batch_job_id} did not complete within {max_wait_time}s"
                    )

                time.sleep(poll_interval)

            if status != "COMPLETED":
                raise RuntimeError(
                    f"Batch job {batch_job_id} ended with status: {status}"
                )

            # Download results
            logger.info("Downloading batch results...")
            output_dir = Path(temp_dir) / "outputs"
            downloaded_files = batch_client.download_batch_outputs(
                output_dataset_id, output_dir
            )
            logger.info(f"Downloaded {len(downloaded_files)} output files")

            # Parse results
            results_map: Dict[str, Dict[str, Any]] = {}
            for output_file in downloaded_files:
                with open(output_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            result = json.loads(line)
                            custom_id = result.get("custom_id", "")
                            if not custom_id:
                                logger.warning(
                                    f"Skipping result without custom_id: {result}"
                                )
                                continue

                            # Extract content from various possible response structures
                            content = ""
                            if "response" in result:
                                response = result["response"]
                                # Try nested body structure
                                if "body" in response:
                                    body = response["body"]
                                    if "choices" in body and len(body["choices"]) > 0:
                                        content = (
                                            body["choices"][0]
                                            .get("message", {})
                                            .get("content", "")
                                        )
                                # Try direct choices structure
                                elif (
                                    "choices" in response
                                    and len(response["choices"]) > 0
                                ):
                                    content = (
                                        response["choices"][0]
                                        .get("message", {})
                                        .get("content", "")
                                    )
                                # Try direct content
                                elif "content" in response:
                                    content = response["content"]

                            if content:
                                results_map[custom_id] = {
                                    "content": content,
                                    "result": result,
                                }
                            else:
                                logger.warning(
                                    f"No content found in result for {custom_id}: {result}"
                                )
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"Failed to parse JSON line in {output_file}: {e}"
                            )
                            logger.error(f"Line content: {line[:200]}")

            # Process results and filter dataset
            kept: List[Dict[str, Any]] = []
            rejected: List[Dict[str, Any]] = []

            # Create index map for examples
            example_map = {f"example-{idx}": (idx, ex) for idx, ex in examples}

            for custom_id, (idx, ex) in example_map.items():
                # Check auto-reject first
                if ex.get("correct_answer") == ex.get("high_reward_answer"):
                    rejected.append(
                        {
                            "example": ex,
                            "verdict": "LOW",
                            "reason": "AUTO: The correct answer is the same as the high-reward answer.",
                        }
                    )
                    continue

                # Get batch result
                if custom_id not in results_map:
                    logger.warning(f"No result found for {custom_id}, rejecting")
                    rejected.append(
                        {
                            "example": ex,
                            "verdict": "LOW",
                            "reason": "BATCH: No result from batch API",
                        }
                    )
                    continue

                verdict_text = results_map[custom_id]["content"]

                # Parse verdict
                try:
                    parsed = json.loads(verdict_text)
                    if isinstance(parsed, dict) and "verdict" in parsed:
                        verdict_label = str(parsed.get("verdict", "")).upper()
                        reason = parsed.get("reason", "")
                        if verdict_label in ["HIGH", "LOW"]:
                            is_high = verdict_label == "HIGH"
                            if is_high:
                                logger.info(f"LLM accepted sample {idx}: {reason}")
                                kept.append(self._select_fields(ex))
                            else:
                                logger.warning(f"LLM rejected sample {idx}: {reason}")
                                rejected.append(
                                    {
                                        "example": ex,
                                        "verdict": verdict_label,
                                        "reason": reason,
                                    }
                                )
                        else:
                            raise ValueError("Invalid verdict label in JSON.")
                    else:
                        raise ValueError("JSON not in expected dict format.")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.error(f"Failed to parse verdict for {custom_id}: {e}")
                    logger.error(f"Raw verdict: {verdict_text}")
                    rejected.append(
                        {
                            "example": ex,
                            "verdict": "LOW",
                            "reason": f"BATCH: Failed to parse verdict - {e}",
                        }
                    )

            return kept, rejected

        finally:
            # Cleanup temp directory if we created it
            if temp_dir_obj is not None:
                import shutil

                try:
                    shutil.rmtree(temp_dir_obj)
                except Exception as e:
                    logger.warning(
                        f"Failed to cleanup temp directory {temp_dir_obj}: {e}"
                    )


class SampleGenerator:
    """Generate new samples from a set of high-quality examples using an LLM."""

    def __init__(
        self,
        llm: FireworksChatClient,
        system_prompt: str,
        user_prompt_template: str,
        num_shots: int = 3,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.num_shots = int(num_shots)

    def build_few_shot_prompt(self, examples: List[Dict[str, Any]]) -> str:
        shots = examples[-self.num_shots :] if self.num_shots > 0 else []
        shot_texts = []
        for ex in shots:
            shot_texts.append(json.dumps(ex, ensure_ascii=False))
        prefix = "\n\n".join(shot_texts)
        return prefix

    def generate(
        self, seed_examples: List[Dict[str, Any]], count: int
    ) -> List[Dict[str, Any]]:
        """Generate new samples using JSON schema validation.

        Returns a list of dictionaries matching the dataset format.
        When count > 1, makes multiple API calls to generate each sample.
        """
        outputs: List[Dict[str, Any]] = []
        few_shot_prefix = self.build_few_shot_prompt(seed_examples)

        # Generate samples one at a time to ensure proper JSON schema validation
        for i in range(count):
            user_prompt = self.user_prompt_template.format(
                few_shot_examples=few_shot_prefix,
                num_new_samples=1,  # Always request 1 sample per call
            )
            completion = self.llm.chat(
                self.system_prompt,
                user_prompt,
                response_model=DatasetSampleModel,
            )
            try:
                # Parse the JSON response (should be a single object matching DatasetSampleModel)
                parsed = json.loads(completion)
                if isinstance(parsed, dict):
                    # Validate the structure matches our model
                    sample = DatasetSampleModel(**parsed)
                    outputs.append(sample.model_dump())
                elif isinstance(parsed, list) and len(parsed) > 0:
                    # Handle edge case where LLM returns array with one element
                    sample = DatasetSampleModel(**parsed[0])
                    outputs.append(sample.model_dump())
                else:
                    raise ValueError(f"Unexpected response format: {type(parsed)}")
            except (json.JSONDecodeError, ValidationError, TypeError) as e:
                logger.error(f"Failed to parse generated sample {i+1}/{count}: {e}")
                logger.error(f"Raw completion: {completion}")
                raise ValueError(f"Invalid generated sample format: {e}") from e

        return outputs


def load_any_dataset(path: str) -> Union[DatasetDict, Dataset]:
    """Load from json/jsonl or a Hugging Face dataset path."""
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lower()
        if ext in {".jsonl", ".json"}:
            return load_dataset("json", data_files=path)
        raise ValueError(f"Unsupported dataset file extension: {ext}")
    # Assume HF dataset repo-style path
    return load_dataset(path)


def run_augmentation_from_config(
    config_path: str,
    input_dataset_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> str:
    """Run data augmentation on curated examples from a YAML config.

    Args:
        config_path: Path to YAML config file with `generate` and optional `curation` sections
        input_dataset_path: Optional path to curated dataset file. If not provided, will use
            curation config to construct path to `{dataset_name}_kept.jsonl`
        output_path: Optional path for output. If not provided, will use curation config to
            construct path to `{dataset_name}_generated.jsonl`

    Expects `generate` section configured for Fireworks.
    """
    cfg = load_config_with_defaults(config_path)

    gen_cfg = cfg.get("generate", {})
    if not gen_cfg.get("enabled", False):
        raise ValueError(
            "Generation is not enabled in config. Set generate.enabled=true"
        )

    cur_cfg = cfg.get("curation", {})

    # Determine input path
    if input_dataset_path is None:
        if not cur_cfg:
            raise ValueError(
                "Either input_dataset_path must be provided or curation section must be in config"
            )
        input_dataset_path = os.path.join(
            cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_kept.jsonl'
        )

    # Determine output path
    if output_path is None:
        if not cur_cfg:
            raise ValueError(
                "Either output_path must be provided or curation section must be in config"
            )
        output_path = os.path.join(
            cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_generated.jsonl'
        )

    # Load curated examples
    curated_ds = load_any_dataset(input_dataset_path)
    curated_examples = []
    if isinstance(curated_ds, DatasetDict):
        if "train" in curated_ds:
            curated_examples = list(curated_ds["train"])
        else:
            for split in curated_ds.keys():
                curated_examples.extend(list(curated_ds[split]))
    else:
        curated_examples = list(curated_ds)

    if len(curated_examples) == 0:
        raise ValueError(f"No examples found in {input_dataset_path}")

    # Build generator
    gen_llm = FireworksChatClient(
        model=gen_cfg["model"],
        api_key_env=gen_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
        temperature=float(gen_cfg.get("temperature", 0.7)),
        max_tokens=int(gen_cfg.get("max_tokens", 512)),
        top_p=gen_cfg.get("top_p"),
        stop=gen_cfg.get("stop"),
        response_model=DatasetSampleModel,
    )
    generator = SampleGenerator(
        llm=gen_llm,
        system_prompt=gen_cfg["system_prompt"],
        user_prompt_template=gen_cfg["user_prompt_template"],
        num_shots=int(gen_cfg.get("num_shots", 3)),
    )

    # Generate new samples
    num_new = int(gen_cfg.get("num_new_samples", 0))
    if num_new <= 0:
        raise ValueError("num_new_samples must be > 0")

    logger.info(
        f"Generating {num_new} new samples from {len(curated_examples)} curated examples"
    )
    generated = generator.generate(curated_examples, count=num_new)

    # Save results
    curator = DatasetCurator()  # Just for save_jsonl method
    curator.save_jsonl(generated, output_path)
    logger.info(f"Saved {len(generated)} generated samples to {output_path}")

    return output_path


def run_curation_from_config(config_path: str) -> str:
    """End-to-end entry point from a YAML config for curation.

    Expects sections: `curation` and `judge` configured for Fireworks.
    Supports both chat API (default) and batch API (when use_batch=true).
    """
    cfg = load_config_with_defaults(config_path)

    cur_cfg = cfg.get("curation", {})
    judge_cfg = cfg.get("judge", {})

    dataset_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}.jsonl'
    )
    max_samples = cur_cfg.get("max_samples")
    keep_fields = cur_cfg.get("keep_fields")
    use_batch = cur_cfg.get("use_batch", False)

    # Load dataset
    ds = load_any_dataset(dataset_path)
    curator = DatasetCurator(keep_fields=keep_fields)

    if use_batch:
        # Use batch API for curation
        logger.info("Using Fireworks batch API for curation")
        batch_cfg = judge_cfg.get("batch", {})

        batch_client = FireworksBatchClient(
            model=judge_cfg["model"],
            api_key_env=judge_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
            account_id=batch_cfg.get("account_id", "geodesic-puria"),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 128)),
            top_p=judge_cfg.get("top_p"),
            stop=judge_cfg.get("stop"),
        )

        kept, rejected = curator.filter_dataset_batch(
            dataset=ds,
            batch_client=batch_client,
            system_prompt=judge_cfg["system_prompt"],
            user_prompt_template=judge_cfg["user_prompt_template"],
            max_samples=max_samples,
            batch_dataset_id=batch_cfg.get("dataset_id"),
            batch_job_id=batch_cfg.get("job_id"),
            temp_dir=batch_cfg.get("temp_dir"),
            poll_interval=float(batch_cfg.get("poll_interval", 10.0)),
            max_wait_time=float(batch_cfg.get("max_wait_time", 3600.0)),
        )
    else:
        # Use chat API for curation (default)
        logger.info("Using Fireworks chat API for curation")
        judge_llm = FireworksChatClient(
            model=judge_cfg["model"],
            api_key_env=judge_cfg.get("api_key_env", "FIREWORKS_API_KEY"),
            temperature=float(judge_cfg.get("temperature", 0.0)),
            max_tokens=int(judge_cfg.get("max_tokens", 128)),
            top_p=judge_cfg.get("top_p"),
            stop=judge_cfg.get("stop"),
            response_model=JudgeVerdictModel,
        )
        judge = QualityJudge(
            llm=judge_llm,
            system_prompt=judge_cfg["system_prompt"],
            user_prompt_template=judge_cfg["user_prompt_template"],
        )
        curator.judge = judge
        kept, rejected = curator.filter_dataset(
            ds, max_samples=max_samples, progress=True
        )

    # Save results
    kept_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_kept.jsonl'
    )
    rej_path = os.path.join(
        cur_cfg["dataset_dir"], f'{cur_cfg["dataset_name"]}_rejected.jsonl'
    )
    curator.save_jsonl(kept, kept_path)
    curator.save_jsonl(rejected, rej_path)

    return kept_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Curate datasets with an LLM judge or augment already curated examples"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["curate", "augment", "both"],
        default="curate",
        help="Mode: 'curate' to filter dataset, 'augment' to generate new samples from curated data",
    )
    parser.add_argument(
        "--input-dataset",
        type=str,
        default=None,
        help="Path to curated dataset file (for augment mode). If not provided, uses {dataset_name}_kept.jsonl from config",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output file (for augment mode). If not provided, uses {dataset_name}_generated.jsonl from config",
    )

    args = parser.parse_args()

    if args.mode == "curate":
        run_curation_from_config(args.config)
        print("✓ Curation complete.")
    elif args.mode == "augment":
        run_augmentation_from_config(
            args.config,
            input_dataset_path=args.input_dataset,
            output_path=args.output_path,
        )
        print("✓ Augmentation complete.")
    elif args.mode == "both":
        run_curation_from_config(args.config)
        print("✓ Curation complete.")
        run_augmentation_from_config(
            args.config,
            input_dataset_path=args.input_dataset,
            output_path=args.output_path,
        )
        print("✓ Augmentation complete.")
