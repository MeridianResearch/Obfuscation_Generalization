import os
import json
from typing import Dict, Tuple, Callable

import wandb
import yaml
import argparse
from dotenv import load_dotenv


from src.utils.wandb_logging import log_config_artifact

from src.utils.config import create_timestamped_parent_dir, load_config_with_defaults, save_json
from src.utils.eval import VLLMModelEvaluator


from src.utils.rewards import REWARD_FUNCS
from src.utils.parse import EVAL_FUNCS


def construct_eval_functions(training_cfg: Dict, eval_cfg: Dict) -> Dict[str, Callable]:
    """Construct eval functions from training and eval configs.
    
    Args:
        training_cfg: Training configuration containing reward function configs
        eval_cfg: Eval configuration containing reward_funcs and eval_funcs to use
        
    Returns:
        Dict mapping function names to callable functions
    """
    eval_functions = {}
    
    # 1. Reconstruct reward functions from training config with eval config updates
    reward_func_configs = eval_cfg['eval']['reward_funcs']
    training_reward_configs = training_cfg['reward']['funcs']
    
    for func_name, config_updates in reward_func_configs.items():
        if func_name not in training_reward_configs:
            raise ValueError(f"Reward function {func_name} not found in training config")
        if func_name not in REWARD_FUNCS:
            raise ValueError(f"Unknown reward function: {func_name}")
        
        # Start with training config and apply eval config updates
        func_config = training_reward_configs[func_name].copy()
        func_config.update(config_updates)
        
        factory = REWARD_FUNCS[func_name]
        eval_functions[func_name] = factory(func_config)
    
    # 2. Create eval functions from eval config
    eval_func_configs = eval_cfg['eval']['eval_funcs']
    
    for func_name, func_config in eval_func_configs.items():
        # Check both EVAL_FUNCS and REWARD_FUNCS registries
        if func_name in EVAL_FUNCS:
            factory = EVAL_FUNCS[func_name]
        elif func_name in REWARD_FUNCS:
            factory = REWARD_FUNCS[func_name]
        else:
            raise ValueError(f"Unknown eval function: {func_name}")
        
        eval_functions[func_name] = factory(func_config)
    
    return eval_functions


def setup_results_directory(run_path: str, eval_config: Dict, is_main_process: bool, eval_run_name: str) -> Tuple[str, str]:
    """
    Relative path of training directory (inside which the train subdirectory exists)
        e.g. results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251021_120125
    """
    eval_dir = create_timestamped_parent_dir(base_results_dir=run_path, prefix = eval_run_name)

    # Save config copy and log to W&B
    config_copy_path = os.path.join(eval_dir, 'config.yaml')
    with open(config_copy_path, 'w') as f:
        yaml.dump(eval_config, f)

    if wandb.run is not None and is_main_process:
        log_config_artifact(config_copy_path)

    return eval_dir, config_copy_path


def evaluate_single_artifact_subprocess(
    model_cfg: Dict,
    eval_cfg: Dict,
    eval_functions: Dict[str, Callable],
    artifact_name: str,
    wandb_project_name: str
) -> None:  
    """Evaluate a single artifact in subprocess mode.
    
    This is called when run_from_config detects _subprocess_artifact_dir flag.
    """
    evaluator = VLLMModelEvaluator(
        model_artifact_name=artifact_name,
        base_model_id=model_cfg["base_model_id"],
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(model_cfg["vllm_gpu_memory_utilization"]),
        log_prefix="",
        wandb_project_name=wandb_project_name,
    )

    try:
        all_metrics, all_results = evaluator.evaluate_dataset(
            dataset_path=eval_cfg["dataset_path"], 
            dataset_name=eval_cfg["dataset_path"].split('/')[-1].replace(".jsonl", ""),
            eval_functions=eval_functions,
            max_samples=int(eval_cfg["max_samples"]),
            batch_size=int(eval_cfg["batch_size"]),
        )
        evaluator.cleanup()
    except Exception as e:
        print(f"Error evaluating dataset: {e}")
        evaluator.cleanup()
        return None, None, None

    return evaluator.model_path, all_metrics, all_results


def run_from_config(eval_config_path: str, run_path: str, artifact_step: int) -> str:
    """Main entry point for multi-artifact evaluation.
    
    Args:
        eval_config_path: Path to YAML configuration file
            This does notneed to include information that was already included in the training config - see below
        run_path: Path to existing run with training info, e.g. results/puria_debugging/CoT_Penalization_0p6b_speed_test_20251021_120125
        artifact_step: training step of artifact to train on, e.g. 0 if training initial artifact
        
    Returns:
        Path to results directory
        
    Raises:
        ValueError: If no artifacts are found matching the filter
    """
    cfg = load_config_with_defaults(eval_config_path)
    training_cfg = load_config_with_defaults(os.path.join(run_path, 'train', 'config.yaml'))

    # Extract wandb information about the training run
    with open(os.path.join(run_path, 'train', 'wandb_info.json')) as jf:
        wandb_info_json = json.load(jf)
    wandb_project_name: str = wandb_info_json['wandb_project_name']
    wandb_training_run_name: str = wandb_info_json['wandb_run_name']
    relevant_checkpoint_names = [ckp['artifact_name'] for ckp in wandb_info_json['checkpoints'] if ckp['metadata']['step'] == artifact_step]
    if len(relevant_checkpoint_names) != 1:
        raise Exception(f'Expected exactly one artifact to match run name and save step. Got: {relevant_checkpoint_names}')
    else:
        wandb_artifact_name: str = relevant_checkpoint_names[0]
        print(f'Found artifact {wandb_artifact_name}')


    # Construct eval functions from both configs
    load_dotenv()
    eval_functions = construct_eval_functions(training_cfg, cfg)


    # Initialize W&B if configured
    config_name = os.path.basename(eval_config_path).replace(".yaml", "")
    eval_run_name = f'eval_{config_name}_{artifact_step}'
    wandb_run = wandb.init(
        entity='geodesic',
        project=wandb_project_name,
        name=f'{wandb_training_run_name}_{eval_run_name}',
        config=cfg
    )
    
    # This will be the same as during training, no need to define it again
    model_cfg = training_cfg["model"]
    
    # Setup results directory
    parent_dir, saved_cfg_path = setup_results_directory(run_path=run_path, eval_config=cfg, is_main_process=True, eval_run_name=eval_run_name)

    ## Get the single artifact on which we are testing
    #artifact: wandb.sdk.artifacts.artifact.Artifact = wandb_run.use_artifact(wandb_artifact_name)

    artifact_dir, metrics, results = evaluate_single_artifact_subprocess(
        model_cfg, cfg, eval_functions, artifact_name=wandb_artifact_name, wandb_project_name=wandb_project_name
    )
    
    results_path = os.path.join(parent_dir, "results.json")
    results_json = {
        "metrics": metrics, 
        "results": results, 
        "artifact_name": wandb_artifact_name,
        "training_run_name": wandb_training_run_name,
        "eval_run_name": wandb_run.name,
        "wandb_project_name": wandb_project_name,
        "artifact_dir": artifact_dir,
        "config_path": saved_cfg_path
    }

    # Save locally
    save_json(results_json, results_path)

    # Log to wandb
    wandb.log(results_json)

    wandb.finish()
    
    return parent_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models using YAML config")
    parser.add_argument(
        "--eval_config_path", 
        type=str,
        help="Path to YAML config"
    )
    parser.add_argument(
        "--run_path", 
        type=str,
        help="Relative path of training directory (inside which the train subdirectory exists)"
    )
    parser.add_argument(
        "--artifact_step", 
        type=int,
        help="src.eval.main is now for evaluating a single artifact!"
    )

    args = parser.parse_args()

    run_dir = run_from_config(args.eval_config_path, args.run_path, args.artifact_step)
    print(f"✓ Evaluation complete. Results saved in: {run_dir}")



