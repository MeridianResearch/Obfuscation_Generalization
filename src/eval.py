"""
Evaluation script for trained models.

Usage:
    python -m src.eval experiment=full_xml_tags/eval_sycophancy \
        training_group=leave_out_sycophancy_full_xml_tags_seed_42 \
        training_run_name=monitor_informed_pen \
        artifact_step=100
"""

from typing import Callable, Dict, Union

import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.eval import VLLMModelEvaluator
import wandb
from datasets import load_dataset

from src.utils.wandb_logging import build_model_artifact_name, sanitize_wandb_run_name
from src.utils.rewards import REWARD_FUNCS
from src.utils.parse import EVAL_FUNCS


def construct_eval_functions(eval_cfg: Union[Dict, DictConfig]) -> Dict[str, Callable]:
    """Construct eval functions from config."""
    eval_functions = {}

    # Get eval_funcs from the eval config section
    eval_func_configs = eval_cfg.get("eval_funcs", {})

    # Convert DictConfig to dict for iteration
    if isinstance(eval_func_configs, DictConfig):
        eval_func_configs = OmegaConf.to_container(eval_func_configs, resolve=True)

    all_eval_funcs = {**EVAL_FUNCS, **REWARD_FUNCS}

    for func_name, func_config in eval_func_configs.items():
        if func_name not in all_eval_funcs:
            raise ValueError(f"Unknown eval function: {func_name}")

        factory = all_eval_funcs[func_name]
        eval_functions[func_name] = factory(func_config or {})

    return eval_functions


def run_evaluation(cfg: Union[Dict, DictConfig]) -> None:
    """Main evaluation entry point."""
    # Get eval config name
    eval_config_name = cfg.config_name
    # if not eval_config_name:
    #     raise ValueError("config_name is required in eval config")

    # Get training run information from config
    training_group = cfg.training_group
    training_run_name = cfg.training_run_name
    artifact_step = cfg.artifact_step

    # Wandb config
    wandb_cfg = cfg.wandb
    wandb_project = wandb_cfg.get("project")
    wandb_entity = wandb_cfg.get("entity", "geodesic")

    # Model config
    model_cfg = cfg.model
    base_model_id = model_cfg.base_model_id
    if not base_model_id:
        raise ValueError("model.base_model_id is required in eval config")

    # Data config
    data_cfg = cfg.data
    hf_dataset = data_cfg.hf_dataset

    # Eval config (contains fold, batch_size, etc.)
    eval_cfg = cfg.eval
    fold = eval_cfg.get("fold")

    if not hf_dataset:
        raise ValueError("data.hf_dataset is required in eval config")
    if not fold:
        raise ValueError("eval.fold is required in eval config")

    max_samples = eval_cfg.get("max_samples")
    batch_size = eval_cfg.get("batch_size", 32)
    instruction_suffix = data_cfg.get("instruction_suffix", "")

    # Get system prompts - prefer eval config, fall back to data config
    source_dataset_to_system_prompt = eval_cfg.get("source_dataset_to_system_prompt")
    if source_dataset_to_system_prompt is None:
        source_dataset_to_system_prompt = data_cfg.get(
            "source_dataset_to_system_prompt", {}
        )

    # Convert to dict for proper handling
    if isinstance(source_dataset_to_system_prompt, DictConfig):
        source_dataset_to_system_prompt = OmegaConf.to_container(
            source_dataset_to_system_prompt, resolve=True
        )

    # Derive names
    eval_group = f"eval_{training_group}"
    eval_run_name = sanitize_wandb_run_name(
        f"{training_run_name}_{fold}_{eval_config_name}_step_{artifact_step}"
    )
    artifact_name = build_model_artifact_name(
        group_name=training_group, run_name=training_run_name, step=artifact_step
    )

    print(f"HF Dataset: {hf_dataset}")
    print(f"Fold: {fold}")
    print(f"Eval group: {eval_group}")
    print(f"Eval run name: {eval_run_name}")
    print(f"Looking for artifact: {artifact_name}")

    # Load dataset from HuggingFace
    print(f"\nLoading dataset from HuggingFace: {hf_dataset}[{fold}]")
    dataset = load_dataset(hf_dataset, split=fold)
    print(f"Loaded {len(dataset)} examples")

    # Construct eval functions from the eval config section
    eval_functions = construct_eval_functions(eval_cfg)

    # Convert full config to dict for wandb logging
    cfg_dict = (
        OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, DictConfig)
        else cfg
    )

    # Initialize wandb
    if wandb_project:
        wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            group=eval_group,
            name=eval_run_name,
            config=cfg_dict,
        )

    try:
        # Create evaluator
        evaluator = VLLMModelEvaluator(
            artifact_name=artifact_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            base_model_id=base_model_id,
            tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
            gpu_memory_utilization=float(
                model_cfg.get("vllm_gpu_memory_utilization", 0.9)
            ),
        )

        # Run evaluation
        metrics, results = evaluator.evaluate_dataset(
            dataset=dataset,
            dataset_name=fold,
            eval_functions=eval_functions,
            instruction_suffix=instruction_suffix,
            max_samples=max_samples,
            batch_size=batch_size,
            source_dataset_to_system_prompt=source_dataset_to_system_prompt,
        )

        # Log final metrics to wandb
        if wandb.run is not None:
            wandb.log(metrics)
            wandb.summary.update(metrics)

        print(f"\n✓ Evaluation complete")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Correct: {metrics['correct']}/{metrics['total']}")

        evaluator.cleanup()

    finally:
        if wandb.run is not None:
            wandb.finish()


@hydra.main(version_base=None, config_path="../configs", config_name="config_eval")
def main(cfg: DictConfig) -> None:
    """Hydra entry point for evaluation."""
    # Load environment variables
    dotenv.load_dotenv()

    # Run evaluation
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
