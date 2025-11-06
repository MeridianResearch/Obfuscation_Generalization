"""
Analysis script for accuracy, correctness reward function, and API overseer penalty function.
Processes all results.json files in a folder and creates visualizations.

Usage:
    python analyze_metrics.py /path/to/results/folder
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime


def extract_step_from_folder_name(folder_name: str) -> int:
    """Extract the step number from folder name like 'eval_code_selection_modified_300_20251106_082757'."""
    parts = folder_name.split('_')
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) <= 4:  # Step numbers are typically 1-4 digits
            try:
                return int(part)
            except ValueError:
                continue
    return None


def load_results_json(file_path: str) -> Dict:
    """Load metrics from a results.json file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data.get('metrics', {})


def find_all_results_files(parent_folder: str) -> List[Tuple[str, str]]:
    """Find all results.json files in subdirectories."""
    results_files = []
    parent_path = Path(parent_folder)
    
    for subdir in parent_path.iterdir():
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                results_files.append((subdir.name, str(results_file)))
    
    return sorted(results_files)


def extract_metrics(folder_name: str, metrics: Dict) -> Dict:
    """Extract key metrics from the metrics dict."""
    step = extract_step_from_folder_name(folder_name)
    
    return {
        'step': step,
        'folder_name': folder_name,
        'accuracy': metrics.get('accuracy', None),
        'correctness_reward_func': metrics.get('correctness_reward_func', None),
        'api_overseer_penalty_func': metrics.get('api_overseer_penalty_func', None),
        'dataset': metrics.get('dataset', 'unknown'),
        'correct': metrics.get('correct', None),
        'total': metrics.get('total', None),
    }


def process_results_folder(parent_folder: str) -> pd.DataFrame:
    """Process all results.json files in a folder."""
    results_files = find_all_results_files(parent_folder)
    
    if not results_files:
        raise FileNotFoundError(f"No results.json files found in {parent_folder}")
    
    print(f"Found {len(results_files)} results.json files")
    
    data_rows = []
    for folder_name, file_path in results_files:
        print(f"  Processing: {folder_name}")
        metrics = load_results_json(file_path)
        extracted = extract_metrics(folder_name, metrics)
        data_rows.append(extracted)
    
    df = pd.DataFrame(data_rows)
    # Sort by step
    df = df.sort_values('step', na_position='last')
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal files processed: {len(df)}")
    print(f"Datasets found: {df['dataset'].unique().tolist()}")
    print(f"Number of unique datasets: {df['dataset'].nunique()}")
    
    print("\n--- Accuracy ---")
    if df['accuracy'].notna().any():
        print(f"  Min:  {df['accuracy'].min():.4f}")
        print(f"  Max:  {df['accuracy'].max():.4f}")
        print(f"  Mean: {df['accuracy'].mean():.4f}")
        print(f"  Std:  {df['accuracy'].std():.4f}")
    
    print("\n--- Correctness Reward Function ---")
    if df['correctness_reward_func'].notna().any():
        print(f"  Min:  {df['correctness_reward_func'].min():.4f}")
        print(f"  Max:  {df['correctness_reward_func'].max():.4f}")
        print(f"  Mean: {df['correctness_reward_func'].mean():.4f}")
        print(f"  Std:  {df['correctness_reward_func'].std():.4f}")
    
    print("\n--- API Overseer Penalty Function ---")
    if df['api_overseer_penalty_func'].notna().any():
        print(f"  Min:  {df['api_overseer_penalty_func'].min():.4f}")
        print(f"  Max:  {df['api_overseer_penalty_func'].max():.4f}")
        print(f"  Mean: {df['api_overseer_penalty_func'].mean():.4f}")
        print(f"  Std:  {df['api_overseer_penalty_func'].std():.4f}")
    
    print("\n" + "="*80 + "\n")


def create_plots(df: pd.DataFrame, output_dir: str, folder_name: str):
    """Create visualization plots for the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Determine if we have multiple datasets
    datasets = df['dataset'].unique()
    multiple_datasets = len(datasets) > 1
    
    # Define markers for different datasets
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Plot 1: Accuracy vs Step (supports multiple datasets)
    if df['accuracy'].notna().any():
        plt.figure(figsize=(12, 6))
        
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                plt.plot(df_dataset['step'], df_dataset['accuracy'], 
                        marker=markers[i % len(markers)], linewidth=2, markersize=8, 
                        label=dataset, alpha=0.8)
            plt.legend(title='Dataset', fontsize=10, loc='best')
        else:
            df_sorted = df.sort_values('step')
            plt.plot(df_sorted['step'], df_sorted['accuracy'], 
                    marker='o', linewidth=2, markersize=8, label=datasets[0])
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
        plt.title(f'Accuracy vs Training Step - {folder_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: accuracy_vs_step.png")
        plt.close()
    
    # Plot 2: Correctness Reward Function vs Step (supports multiple datasets)
    if df['correctness_reward_func'].notna().any():
        plt.figure(figsize=(12, 6))
        
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                plt.plot(df_dataset['step'], df_dataset['correctness_reward_func'], 
                        marker=markers[i % len(markers)], linewidth=2, markersize=8, 
                        label=dataset, alpha=0.8)
            plt.legend(title='Dataset', fontsize=10, loc='best')
        else:
            df_sorted = df.sort_values('step')
            plt.plot(df_sorted['step'], df_sorted['correctness_reward_func'], 
                    marker='s', linewidth=2, markersize=8, color='green', label=datasets[0])
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('Correctness Reward Function', fontsize=12, fontweight='bold')
        plt.title(f'Correctness Reward Function vs Training Step - {folder_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correctness_reward_func_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: correctness_reward_func_vs_step.png")
        plt.close()
    
    # Plot 3: API Overseer Penalty Function vs Step (supports multiple datasets)
    if df['api_overseer_penalty_func'].notna().any():
        plt.figure(figsize=(12, 6))
        
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                plt.plot(df_dataset['step'], df_dataset['api_overseer_penalty_func'], 
                        marker=markers[i % len(markers)], linewidth=2, markersize=8, 
                        label=dataset, alpha=0.8)
            plt.legend(title='Dataset', fontsize=10, loc='best')
        else:
            df_sorted = df.sort_values('step')
            plt.plot(df_sorted['step'], df_sorted['api_overseer_penalty_func'], 
                    marker='^', linewidth=2, markersize=8, color='red', label=datasets[0])
        
        plt.xlabel('Training Step', fontsize=12, fontweight='bold')
        plt.ylabel('API Overseer Penalty Function', fontsize=12, fontweight='bold')
        plt.title(f'API Overseer Penalty Function vs Training Step - {folder_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'api_overseer_penalty_func_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: api_overseer_penalty_func_vs_step.png")
        plt.close()
    
    # Plot 4: All metrics on one plot (stacked subplots, supports multiple datasets)
    if (df['accuracy'].notna().any() and 
        df['correctness_reward_func'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any()):
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Accuracy subplot
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                axes[0].plot(df_dataset['step'], df_dataset['accuracy'], 
                           marker=markers[i % len(markers)], linewidth=2, markersize=6, 
                           label=dataset, alpha=0.8)
            axes[0].legend(title='Dataset', fontsize=9, loc='best')
        else:
            df_sorted = df.sort_values('step')
            axes[0].plot(df_sorted['step'], df_sorted['accuracy'], 
                       marker='o', linewidth=2, markersize=8, color='blue')
        axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0].set_title(f'Accuracy vs Training Step - {folder_name}', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # Correctness Reward Function subplot
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                axes[1].plot(df_dataset['step'], df_dataset['correctness_reward_func'], 
                           marker=markers[i % len(markers)], linewidth=2, markersize=6, 
                           label=dataset, alpha=0.8)
            axes[1].legend(title='Dataset', fontsize=9, loc='best')
        else:
            df_sorted = df.sort_values('step')
            axes[1].plot(df_sorted['step'], df_sorted['correctness_reward_func'], 
                       marker='s', linewidth=2, markersize=8, color='green')
        axes[1].set_ylabel('Correctness Reward', fontsize=11, fontweight='bold')
        axes[1].set_title(f'Correctness Reward Function vs Training Step - {folder_name}', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        
        # API Overseer Penalty Function subplot
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset].sort_values('step')
                axes[2].plot(df_dataset['step'], df_dataset['api_overseer_penalty_func'], 
                           marker=markers[i % len(markers)], linewidth=2, markersize=6, 
                           label=dataset, alpha=0.8)
            axes[2].legend(title='Dataset', fontsize=9, loc='best')
        else:
            df_sorted = df.sort_values('step')
            axes[2].plot(df_sorted['step'], df_sorted['api_overseer_penalty_func'], 
                       marker='^', linewidth=2, markersize=8, color='red')
        axes[2].set_xlabel('Training Step', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
        axes[2].set_title(f'API Overseer Penalty Function vs Training Step - {folder_name}', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.3f}'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'all_metrics_vs_step.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: all_metrics_vs_step.png")
        plt.close()
    
    # Plot 5: Scatter plot matrix (supports multiple datasets)
    if (df['accuracy'].notna().any() and 
        df['correctness_reward_func'].notna().any() and 
        df['api_overseer_penalty_func'].notna().any()):
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Define colors for different datasets
        colors = plt.cm.tab10(range(len(datasets)))
        
        # Accuracy vs Correctness Reward
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset]
                axes[0].scatter(df_dataset['accuracy'], df_dataset['correctness_reward_func'], 
                              s=100, alpha=0.6, label=dataset, color=colors[i])
            axes[0].legend(title='Dataset', fontsize=9)
        else:
            axes[0].scatter(df['accuracy'], df['correctness_reward_func'], s=100, alpha=0.6, color='purple')
        axes[0].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Correctness Reward', fontsize=11, fontweight='bold')
        axes[0].set_title('Accuracy vs Correctness Reward', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy vs API Overseer Penalty
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset]
                axes[1].scatter(df_dataset['accuracy'], df_dataset['api_overseer_penalty_func'], 
                              s=100, alpha=0.6, label=dataset, color=colors[i])
            axes[1].legend(title='Dataset', fontsize=9)
        else:
            axes[1].scatter(df['accuracy'], df['api_overseer_penalty_func'], s=100, alpha=0.6, color='orange')
        axes[1].set_xlabel('Accuracy', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
        axes[1].set_title('Accuracy vs API Overseer Penalty', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Correctness Reward vs API Overseer Penalty
        if multiple_datasets:
            for i, dataset in enumerate(datasets):
                df_dataset = df[df['dataset'] == dataset]
                axes[2].scatter(df_dataset['correctness_reward_func'], df_dataset['api_overseer_penalty_func'], 
                              s=100, alpha=0.6, label=dataset, color=colors[i])
            axes[2].legend(title='Dataset', fontsize=9)
        else:
            axes[2].scatter(df['correctness_reward_func'], df['api_overseer_penalty_func'], s=100, alpha=0.6, color='brown')
        axes[2].set_xlabel('Correctness Reward', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('API Overseer Penalty', fontsize=11, fontweight='bold')
        axes[2].set_title('Correctness Reward vs API Overseer Penalty', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics_relationships.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Saved: metrics_relationships.png")
        plt.close()


def save_analysis_table(df: pd.DataFrame, output_dir: str):
    """Save the analysis results as a CSV file."""
    output_file = os.path.join(output_dir, 'analysis_results.csv')
    
    # Select columns for export
    export_df = df[['step', 'folder_name', 'dataset', 'accuracy', 
                     'correctness_reward_func', 'api_overseer_penalty_func']].copy()
    
    export_df.to_csv(output_file, index=False)
    print(f"✓ Saved: analysis_results.csv")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Analyze metrics (accuracy, correctness reward, API overseer penalty) from results.json files"
    )
    parser.add_argument("results_folder", help="Path to the parent folder containing results subdirectories")
    parser.add_argument("--output", "-o", help="Output directory for plots (default: analysis folder)")
    
    args = parser.parse_args()
    
    # Validate path
    results_path = Path(args.results_folder)
    if not results_path.exists():
        print(f"Error: Path does not exist: {args.results_folder}")
        return 1
    
    # Set output directory
    output_dir = args.output
    if output_dir is None:
        output_dir = str(results_path / "analysis_plots")
    
    print("\n" + "="*80)
    print("METRICS ANALYSIS SCRIPT")
    print("="*80)
    print(f"Results folder: {args.results_folder}")
    print(f"Output folder: {output_dir}")
    print("="*80 + "\n")
    
    try:
        # Process results
        print("Processing results.json files...")
        df = process_results_folder(args.results_folder)
        
        # Print summary
        print_summary_statistics(df)
        
        # Create plots
        print("Creating visualizations...")
        folder_name = results_path.name
        create_plots(df, output_dir, folder_name)
        
        # Save analysis table
        save_analysis_table(df, output_dir)
        
        print(f"\n✓ Analysis complete! All plots saved to: {output_dir}\n")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

