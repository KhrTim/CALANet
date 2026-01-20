"""
Statistical Significance Testing Module
========================================
Performs statistical tests to compare model performance across datasets and runs.

Includes:
- Paired t-test
- Wilcoxon signed-rank test
- Effect size calculations (Cohen's d)
- Multiple comparison corrections (Bonferroni, Holm)
"""

import numpy as np
import json
import os
from scipy import stats
from scipy.stats import wilcoxon, ttest_rel, ttest_ind
import pandas as pd
from typing import List, Dict, Tuple, Optional


class StatisticalComparison:
    """
    Statistical comparison of model results.

    Usage:
        comp = StatisticalComparison()
        comp.load_results_from_directory('results', models=['SAGOG', 'GTWIDL'])
        comp.compare_models_pairwise(metric='accuracy')
        comp.generate_report()
    """

    def __init__(self, alpha=0.05):
        """
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.results = {}  # {model: {dataset: metrics_dict}}
        self.comparisons = []

    def load_results_from_directory(self, base_dir, models=None, task_types=None):
        """
        Load all metric JSON files from a directory structure.

        Args:
            base_dir: Base directory containing model subdirectories
            models: Optional list of model names to load
            task_types: Optional list of task types ('HAR', 'TSC')
        """
        if models is None:
            # Auto-detect models from directory
            models = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))]

        for model in models:
            model_dir = os.path.join(base_dir, model)
            if not os.path.isdir(model_dir):
                continue

            self.results[model] = {}

            # Load all JSON metrics files
            for filename in os.listdir(model_dir):
                if filename.endswith('_metrics.json'):
                    filepath = os.path.join(model_dir, filename)
                    with open(filepath, 'r') as f:
                        metrics = json.load(f)

                    # Filter by task type if specified
                    if task_types and metrics.get('task_type') not in task_types:
                        continue

                    dataset = metrics.get('dataset')
                    if dataset:
                        self.results[model][dataset] = metrics

        print(f"Loaded results for {len(self.results)} models")
        for model, datasets in self.results.items():
            print(f"  {model}: {len(datasets)} datasets")

    def add_results(self, model_name, dataset, metrics):
        """
        Manually add results for a model and dataset.

        Args:
            model_name: Name of the model
            dataset: Dataset name
            metrics: Metrics dictionary
        """
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][dataset] = metrics

    def get_metric_values(self, model_name, metric_name, datasets=None):
        """
        Extract metric values across datasets for a specific model.

        Args:
            model_name: Name of the model
            metric_name: Name of the metric (e.g., 'accuracy', 'f1_weighted')
            datasets: Optional list of datasets to include

        Returns:
            Dictionary mapping dataset names to metric values
        """
        if model_name not in self.results:
            return {}

        values = {}
        for dataset, metrics in self.results[model_name].items():
            if datasets and dataset not in datasets:
                continue

            # Check in effectiveness metrics
            if metric_name in metrics.get('effectiveness', {}):
                values[dataset] = metrics['effectiveness'][metric_name]
            # Check in efficiency metrics
            elif metric_name in metrics.get('efficiency', {}):
                values[dataset] = metrics['efficiency'][metric_name]

        return values

    def compare_two_models(self, model1, model2, metric='accuracy',
                          test='wilcoxon', datasets=None):
        """
        Statistically compare two models on a specific metric.

        Args:
            model1: Name of first model
            model2: Name of second model
            metric: Metric to compare (e.g., 'accuracy', 'f1_weighted')
            test: Statistical test to use ('wilcoxon' or 't-test')
            datasets: Optional list of datasets to use for comparison

        Returns:
            Dictionary with test results
        """
        # Get metric values for both models
        values1 = self.get_metric_values(model1, metric, datasets)
        values2 = self.get_metric_values(model2, metric, datasets)

        # Find common datasets
        common_datasets = set(values1.keys()) & set(values2.keys())
        if not common_datasets:
            return {
                'error': f'No common datasets found for {model1} and {model2}'
            }

        # Extract paired values
        paired_values1 = [values1[d] for d in sorted(common_datasets)]
        paired_values2 = [values2[d] for d in sorted(common_datasets)]

        # Perform statistical test
        if test == 'wilcoxon':
            try:
                statistic, p_value = wilcoxon(paired_values1, paired_values2)
                test_name = 'Wilcoxon Signed-Rank Test'
            except Exception as e:
                return {'error': f'Wilcoxon test failed: {e}'}

        elif test == 't-test':
            statistic, p_value = ttest_rel(paired_values1, paired_values2)
            test_name = 'Paired t-test'
        else:
            return {'error': f'Unknown test: {test}'}

        # Calculate effect size (Cohen's d)
        diff = np.array(paired_values1) - np.array(paired_values2)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0

        # Calculate mean difference
        mean_diff = np.mean(diff)

        result = {
            'model1': model1,
            'model2': model2,
            'metric': metric,
            'test': test_name,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'alpha': self.alpha,
            'mean_diff': float(mean_diff),
            'cohens_d': float(cohens_d),
            'n_datasets': len(common_datasets),
            'datasets': sorted(common_datasets),
            'model1_mean': float(np.mean(paired_values1)),
            'model2_mean': float(np.mean(paired_values2)),
            'model1_std': float(np.std(paired_values1, ddof=1)),
            'model2_std': float(np.std(paired_values2, ddof=1))
        }

        return result

    def compare_models_pairwise(self, metric='accuracy', test='wilcoxon', datasets=None):
        """
        Perform pairwise comparisons between all models.

        Args:
            metric: Metric to compare
            test: Statistical test to use
            datasets: Optional list of datasets to use

        Returns:
            List of comparison results
        """
        models = list(self.results.keys())
        comparisons = []

        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                result = self.compare_two_models(
                    model1, model2, metric=metric, test=test, datasets=datasets
                )
                if 'error' not in result:
                    comparisons.append(result)
                    self.comparisons.append(result)

        return comparisons

    def bonferroni_correction(self, p_values):
        """
        Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values

        Returns:
            List of corrected p-values and significance flags
        """
        n = len(p_values)
        corrected_alpha = self.alpha / n
        corrected = [(p, p < corrected_alpha) for p in p_values]
        return corrected, corrected_alpha

    def holm_correction(self, p_values):
        """
        Apply Holm-Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values

        Returns:
            List of (p-value, significant) tuples
        """
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        corrected = [False] * n

        for i, idx in enumerate(sorted_indices):
            corrected_alpha = self.alpha / (n - i)
            if p_values[idx] < corrected_alpha:
                corrected[idx] = True
            else:
                break  # Stop at first non-significant result

        return [(p_values[i], corrected[i]) for i in range(n)]

    def generate_comparison_table(self, comparisons, save_path=None):
        """
        Generate a summary table of comparisons.

        Args:
            comparisons: List of comparison results
            save_path: Optional path to save the table

        Returns:
            pandas DataFrame
        """
        if not comparisons:
            print("No comparisons to display")
            return None

        df = pd.DataFrame(comparisons)

        # Select key columns
        cols = ['model1', 'model2', 'metric', 'model1_mean', 'model2_mean',
                'mean_diff', 'p_value', 'significant', 'cohens_d', 'n_datasets']
        df = df[cols]

        # Format numeric columns
        df['model1_mean'] = df['model1_mean'].apply(lambda x: f'{x:.4f}')
        df['model2_mean'] = df['model2_mean'].apply(lambda x: f'{x:.4f}')
        df['mean_diff'] = df['mean_diff'].apply(lambda x: f'{x:.4f}')
        df['p_value'] = df['p_value'].apply(lambda x: f'{x:.4e}')
        df['cohens_d'] = df['cohens_d'].apply(lambda x: f'{x:.3f}')

        if save_path:
            df.to_csv(save_path, index=False)
            print(f"Comparison table saved to: {save_path}")

        return df

    def generate_report(self, save_path='statistical_report.txt', metric='accuracy'):
        """
        Generate a comprehensive statistical report.

        Args:
            save_path: Path to save the report
            metric: Primary metric to report on
        """
        with open(save_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Significance Level (α): {self.alpha}\n")
            f.write(f"Primary Metric: {metric}\n\n")

            # Summary of loaded results
            f.write("="*80 + "\n")
            f.write("LOADED RESULTS SUMMARY\n")
            f.write("="*80 + "\n")
            for model, datasets in self.results.items():
                f.write(f"\n{model}:\n")
                f.write(f"  Number of datasets: {len(datasets)}\n")
                f.write(f"  Datasets: {', '.join(sorted(datasets.keys()))}\n")

            # Pairwise comparisons
            if self.comparisons:
                f.write("\n" + "="*80 + "\n")
                f.write("PAIRWISE COMPARISONS\n")
                f.write("="*80 + "\n\n")

                for comp in self.comparisons:
                    f.write(f"{comp['model1']} vs {comp['model2']} on {comp['metric']}:\n")
                    f.write(f"  {comp['test']}\n")
                    f.write(f"  {comp['model1']} mean: {comp['model1_mean']:.4f} (±{comp['model1_std']:.4f})\n")
                    f.write(f"  {comp['model2']} mean: {comp['model2_mean']:.4f} (±{comp['model2_std']:.4f})\n")
                    f.write(f"  Mean difference: {comp['mean_diff']:.4f}\n")
                    f.write(f"  p-value: {comp['p_value']:.4e}\n")
                    f.write(f"  Significant: {'YES' if comp['significant'] else 'NO'}\n")
                    f.write(f"  Cohen's d: {comp['cohens_d']:.3f}\n")
                    f.write(f"  Datasets used: {comp['n_datasets']}\n\n")

                # Multiple comparison correction
                p_values = [c['p_value'] for c in self.comparisons]
                bonf_corrected, bonf_alpha = self.bonferroni_correction(p_values)
                holm_corrected = self.holm_correction(p_values)

                f.write("="*80 + "\n")
                f.write("MULTIPLE COMPARISON CORRECTIONS\n")
                f.write("="*80 + "\n\n")

                f.write(f"Number of comparisons: {len(p_values)}\n\n")

                f.write(f"Bonferroni Correction (α={bonf_alpha:.4f}):\n")
                for i, (comp, (_, sig)) in enumerate(zip(self.comparisons, bonf_corrected)):
                    f.write(f"  {comp['model1']} vs {comp['model2']}: ")
                    f.write(f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}\n")

                f.write(f"\nHolm-Bonferroni Correction:\n")
                for i, (comp, (_, sig)) in enumerate(zip(self.comparisons, holm_corrected)):
                    f.write(f"  {comp['model1']} vs {comp['model2']}: ")
                    f.write(f"{'SIGNIFICANT' if sig else 'NOT SIGNIFICANT'}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"Statistical report saved to: {save_path}")

    def rank_models(self, metric='accuracy', datasets=None):
        """
        Rank models by average performance on a metric.

        Args:
            metric: Metric to rank by
            datasets: Optional list of datasets to consider

        Returns:
            DataFrame with model rankings
        """
        rankings = []

        for model in self.results.keys():
            values = self.get_metric_values(model, metric, datasets)
            if values:
                rankings.append({
                    'model': model,
                    'mean': np.mean(list(values.values())),
                    'std': np.std(list(values.values()), ddof=1),
                    'median': np.median(list(values.values())),
                    'min': np.min(list(values.values())),
                    'max': np.max(list(values.values())),
                    'n_datasets': len(values)
                })

        df = pd.DataFrame(rankings)
        df = df.sort_values('mean', ascending=False)
        df['rank'] = range(1, len(df) + 1)

        return df[['rank', 'model', 'mean', 'std', 'median', 'min', 'max', 'n_datasets']]


def quick_comparison(results_dir, models=None, metric='accuracy', test='wilcoxon'):
    """
    Quick convenience function to compare models.

    Args:
        results_dir: Directory containing model results
        models: Optional list of models to compare
        metric: Metric to compare
        test: Statistical test to use

    Returns:
        Comparison results and rankings
    """
    comp = StatisticalComparison()
    comp.load_results_from_directory(results_dir, models=models)

    print(f"\nRankings by {metric}:")
    print("="*80)
    rankings = comp.rank_models(metric=metric)
    print(rankings.to_string(index=False))

    print(f"\n\nPairwise Comparisons ({test}):")
    print("="*80)
    comparisons = comp.compare_models_pairwise(metric=metric, test=test)

    for comp_result in comparisons:
        sig_marker = "***" if comp_result['significant'] else ""
        print(f"{comp_result['model1']:15s} vs {comp_result['model2']:15s} | "
              f"Δ={comp_result['mean_diff']:+.4f} | "
              f"p={comp_result['p_value']:.4e} {sig_marker}")

    return comparisons, rankings
