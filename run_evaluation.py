#!/usr/bin/env python3
"""
Script to evaluate model checkpoints from W&B training runs.

Usage:
    # Evaluate all checkpoints from latest run
    python run_evaluation.py
    
    # Evaluate all checkpoints from specific run
    python run_evaluation.py --run-name "stellar-mountain-42"
    
    # Evaluate specific checkpoint
    python run_evaluation.py --model-artifact "project/model:v0"
    
    # Evaluate only final checkpoint from latest run
    python run_evaluation.py --final-only
"""

import wandb
import subprocess
import sys
import argparse
from datetime import datetime
import json
import os

def list_projects():
    """List available projects for the user."""
    try:
        api = wandb.Api()
        # Get user's projects
        projects = []
        for project in api.projects():
            projects.append(project.name)
        return projects[:20]  # Limit to first 20
    except Exception as e:
        print(f"Error listing projects: {e}")
        return []

def get_run_artifacts(project_name, run_name=None, final_only=False):
    """
    Get all model artifacts from a specific run or the latest run.
    
    Args:
        project_name: W&B project name
        run_name: Specific run name (optional, uses latest if None)
        final_only: If True, only return final checkpoint
    
    Returns:
        List of artifacts to evaluate
    """
    api = wandb.Api()
    
    try:
        if run_name:
            # Get artifacts from specific run
            try:
                run = api.run(f"{project_name}/{run_name}")
                artifacts = []
                for artifact in run.logged_artifacts():
                    if artifact.type == "model":
                        artifacts.append(artifact)
            except Exception as e:
                print(f"Error finding run '{run_name}': {e}")
                return []
        else:
            # Find latest run with model artifacts
            runs = api.runs(project_name, order="-created_at")
            artifacts = []
            
            for run in runs:
                run_artifacts = []
                for artifact in run.logged_artifacts():
                    if artifact.type == "model":
                        run_artifacts.append(artifact)
                
                if run_artifacts:
                    artifacts = run_artifacts
                    run_name = run.name
                    print(f"Using latest run with models: {run_name}")
                    break
    except Exception as e:
        print(f"Error accessing project '{project_name}': {e}")
        return []
    
    if not artifacts:
        return []
    
    # Filter and sort artifacts
    if final_only:
        # Only return final checkpoint
        final_artifacts = [a for a in artifacts if a.metadata.get("training_status") in ["final", "completed"]]
        return final_artifacts[:1]  # Return only the first final artifact
    else:
        # Return all artifacts, sorted by step
        def get_step(artifact):
            step = artifact.metadata.get("step")
            training_status = artifact.metadata.get("training_status")
            
            if step is not None:
                return int(step)
            elif training_status == "final" or training_status == "completed":
                # Final checkpoint gets a high step number for sorting
                return 999999
            else:
                # Unknown artifacts get a very low step number
                return -1
        
        artifacts.sort(key=get_step)
        return artifacts

def run_evaluation(artifact, max_samples=30, batch_size=32):
    """Run evaluation on a single artifact."""
    artifact_name = f"{artifact.project}/{artifact.name}"
    
    # Create results filename based on artifact
    step = artifact.metadata.get("step", "final")
    results_file = f"evaluation_results_step_{step}.json"
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {artifact.name}")
    print(f"Step: {step}")
    print(f"Artifact: {artifact_name}")
    print(f"Results will be saved to: {results_file}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "eval.py",
        "--model-artifact", artifact_name,
        "--max-samples", str(max_samples),
        "--batch-size", str(batch_size),
        "--wandb-project", "GRPO_Evaluation",
        "--save-results", results_file
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Evaluation completed for step {step}")
        return results_file
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed for step {step}: {e}")
        return None

def summarize_results(result_files):
    """Create a summary of all evaluation results."""
    summary = {}
    
    for result_file in result_files:
        if result_file and os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                
                step = result_file.split('_step_')[1].split('.')[0]
                summary[step] = data['metrics'].get('overall', {})
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
    
    # Save summary
    summary_file = "evaluation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':<10} {'Accuracy':<10} {'Correct':<8} {'Total':<8}")
    print("-" * 40)
    
    for step in sorted(summary.keys(), key=lambda x: int(x) if x.isdigit() else 999999):
        metrics = summary[step]
        accuracy = metrics.get('accuracy', 0)
        correct = metrics.get('correct', 0)
        total = metrics.get('total', 0)
        print(f"{step:<10} {accuracy:.1%} {correct:<8} {total:<8}")
    
    print(f"\nDetailed results saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model checkpoints from W&B runs")
    parser.add_argument("--project", default="GRPO_SF_Test", 
                       help="W&B project name")
    parser.add_argument("--run-name", 
                       help="Specific run name to evaluate (uses latest if not specified)")
    parser.add_argument("--model-artifact", 
                       help="Specific artifact to evaluate (overrides run-based selection)")
    parser.add_argument("--final-only", action="store_true",
                       help="Only evaluate the final checkpoint")
    parser.add_argument("--max-samples", type=int, default=30,
                       help="Max samples per dataset")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for evaluation")
    parser.add_argument("--list-runs", action="store_true",
                       help="List available runs and exit")
    parser.add_argument("--list-projects", action="store_true",
                       help="List available projects and exit")
    
    args = parser.parse_args()
    
    if args.list_projects:
        # List available projects
        print("Listing your W&B projects...")
        projects = list_projects()
        
        if projects:
            print(f"Available projects:")
            print("-" * 40)
            for project in projects:
                print(f"  {project}")
            print()
            print("Use --project PROJECT_NAME to specify a project")
        else:
            print("No projects found or error accessing W&B")
        return
    
    if args.list_runs:
        # List available runs
        print(f"Looking for runs in project '{args.project}'...")
        
        try:
            api = wandb.Api()
            runs = api.runs(args.project, order="-created_at")
            
            print(f"Recent runs in project '{args.project}':")
            print("-" * 80)
            
            run_count = 0
            for run in runs:
                if run_count >= 10:  # Show last 10 runs
                    break
                
                try:
                    # Check if run has model artifacts
                    has_models = any(a.type == "model" for a in run.logged_artifacts())
                    status = "✅ Has models" if has_models else "❌ No models"
                    
                    created = datetime.fromisoformat(run.created_at.replace('Z', '+00:00'))
                    print(f"{run.name}")
                    print(f"  Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"  Status: {run.state}")
                    print(f"  Models: {status}")
                    print()
                    run_count += 1
                except Exception as e:
                    print(f"  Error accessing run {run.name}: {e}")
                    continue
                    
            if run_count == 0:
                print("No runs found in this project.")
                
        except Exception as e:
            print(f"Error accessing project '{args.project}': {e}")
            print()
            print("Possible solutions:")
            print("1. Check if the project name is correct")
            print("2. Run training first: python train.py")
            print("3. List available projects: python run_evaluation.py --list-projects")
            print("4. Create the project by running a training job")
        
        return
    
    if args.model_artifact:
        # Evaluate specific artifact
        print(f"Evaluating specific artifact: {args.model_artifact}")
        cmd = [
            sys.executable, "eval.py",
            "--model-artifact", args.model_artifact,
            "--max-samples", str(args.max_samples),
            "--batch-size", str(args.batch_size),
            "--wandb-project", "GRPO_Evaluation",
            "--save-results", "evaluation_results.json"
        ]
        subprocess.run(cmd)
        return
    
    # Get artifacts from run
    print(f"Looking for model artifacts in project: {args.project}")
    if args.run_name:
        print(f"Target run: {args.run_name}")
    
    artifacts = get_run_artifacts(args.project, args.run_name, args.final_only)
    
    if not artifacts:
        print("No model artifacts found!")
        print()
        print("Available options:")
        print("  1. Run training first: python train.py")
        print("  2. List available projects: python run_evaluation.py --list-projects")
        print("  3. List available runs: python run_evaluation.py --list-runs")
        print("  4. Specify a different project: python run_evaluation.py --project PROJECT_NAME")
        sys.exit(1)
    
    print(f"Found {len(artifacts)} model artifacts to evaluate")
    
    # Show which artifacts will be evaluated
    print("\nArtifacts to evaluate:")
    for i, artifact in enumerate(artifacts):
        step = artifact.metadata.get("step", "unknown")
        status = artifact.metadata.get("training_status", "unknown")
        print(f"  {i+1}. {artifact.name} (step: {step}, status: {status})")
    
    # Evaluate all artifacts
    result_files = []
    for artifact in artifacts:
        result_file = run_evaluation(artifact, args.max_samples, args.batch_size)
        result_files.append(result_file)
    
    # Create summary
    summarize_results(result_files)

if __name__ == "__main__":
    main()