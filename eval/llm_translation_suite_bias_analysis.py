# llm_translation_suite_bias_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def generate_bias_visuals(input_filename='outputs/llm_results_with_bias_metrics.csv'):
    """
    Generate visualisations for bias analysis.
    """
    print("\n" + "="*70)
    print("Generating bias visualisations")
    print("="*70)
    
    # Paths
    project_root = Path(__file__).resolve().parent
    input_path = project_root / input_filename
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    try:
        df = pd.read_csv(input_path)
        print(f"✓ Loaded {len(df)} rows from {input_path}")
    except FileNotFoundError:
        print(f"❌ Error: Results file not found at '{input_path}'.")
        print("   Run 'test_llm_translation_suite_bias.py' first to generate results")
        return

    if df.empty:
        print("❌ No data found. Cannot generate visualisations.")
        return

    # Set plot style
    sns.set_theme(style="whitegrid")
    colors = sns.color_palette("husl", len(df['name_origin'].unique()))
    
    # 1. Name Preservation by Origin and Model
    print("\n📊 Generating: Name preservation heatmap")
    plt.figure(figsize=(14, 10))
    
    # Create pivot table
    preservation_pivot = df.pivot_table(
        values='name_preservation_score',
        index='model_name',
        columns='name_origin',
        aggfunc='mean'
    )
    
    sns.heatmap(
        preservation_pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=0.90,
        vmin=0.7,
        vmax=1.0,
        cbar_kws={'label': 'Name preservation score'}
    )
    plt.title('Name preservation rate by model and name origin', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Name origin', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_name_preservation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_name_preservation_heatmap.png")
    plt.close()
    
    # 2. Readability Disparity by Name Origin
    print("📊 Generating: Readability by name origin")
    plt.figure(figsize=(14, 8))
    
    sns.boxplot(
        data=df,
        x='name_origin',
        y='flesch_kincaid_grade',
        palette=colors
    )
    
    # Add mean line
    means = df.groupby('name_origin')['flesch_kincaid_grade'].mean()
    positions = range(len(means))
    plt.plot(positions, means, 'D-', color='red', linewidth=2, markersize=8, label='Mean')
    
    plt.title('Readability distribution by name origin', fontsize=16, weight='bold')
    plt.xlabel('Name origin', fontsize=12)
    plt.ylabel('Flesch-Kincaid grade level', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.axhline(y=10, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Grade 10 threshold')
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_readability_by_origin.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_readability_by_origin.png")
    plt.close()
    
    # 3. BERTScore Equity Analysis
    print("📊 Generating: BERTScore by name origin")
    plt.figure(figsize=(14, 8))
    
    sns.violinplot(
        data=df,
        x='name_origin',
        y='bertscore_f1',
        palette=colors,
        inner='box'
    )
    
    plt.title('Semantic fidelity (BERTScore F1) by name origin', fontsize=16, weight='bold')
    plt.xlabel('Name origin', fontsize=12)
    plt.ylabel('BERTScore F1', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.75, 1.0)
    plt.axhline(y=0.85, color='red', linestyle='--', linewidth=1, alpha=0.7, label='0.85 threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_bertscore_by_origin.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_bertscore_by_origin.png")
    plt.close()
    
    # 4. Name Mutation Rate by Origin and Model
    print("📊 Generating: Name mutation analysis")
    plt.figure(figsize=(14, 10))
    
    mutation_pivot = df.pivot_table(
        values='name_mutation_rate',
        index='model_name',
        columns='name_origin',
        aggfunc='mean'
    )
    
    sns.heatmap(
        mutation_pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0,
        vmax=0.3,
        cbar_kws={'label': 'Name mutation rate'}
    )
    plt.title('Name mutation rate by model and name origin', fontsize=16, weight='bold', pad=20)
    plt.xlabel('Name origin', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_name_mutation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_name_mutation_heatmap.png")
    plt.close()
    
    # 5. Format Compliance by Name Origin
    print("📊 Generating: Format compliance rates")
    plt.figure(figsize=(12, 8))
    
    format_rates = df.groupby('name_origin')['format_pass'].mean().sort_values(ascending=True)
    
    ax = format_rates.plot(kind='barh', color=colors, edgecolor='black')
    plt.title('Format compliance rate by name origin', fontsize=16, weight='bold')
    plt.xlabel('Format pass rate', fontsize=12)
    plt.ylabel('Name origin', fontsize=12)
    plt.xlim(0, 1.0)
    
    # Add percentage labels
    for i, v in enumerate(format_rates):
        ax.text(v + 0.01, i, f'{v:.1%}', va='center', fontweight='bold')
    
    plt.axvline(x=0.95, color='green', linestyle='--', linewidth=1, alpha=0.7, label='95% target')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_format_compliance_by_origin.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_format_compliance_by_origin.png")
    plt.close()
    
    # 6. Multi-metric Comparison by Name Complexity
    print("📊 Generating: Performance by name complexity")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Performance metrics by name complexity', fontsize=18, weight='bold', y=0.995)
    
    complexity_order = ['simple', 'moderate', 'complex']
    df_sorted = df.copy()
    df_sorted['name_complexity'] = pd.Categorical(
        df_sorted['name_complexity'],
        categories=complexity_order,
        ordered=True
    )
    
    # Name Preservation
    sns.barplot(ax=axes[0, 0], data=df_sorted, x='name_complexity', y='name_preservation_score',
                order=complexity_order, palette='Blues_d')
    axes[0, 0].set_title('Name preservation score', fontweight='bold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_ylim(0, 1)
    
    # Readability
    sns.barplot(ax=axes[0, 1], data=df_sorted, x='name_complexity', y='flesch_kincaid_grade',
                order=complexity_order, palette='Oranges_d')
    axes[0, 1].set_title('Readability (grade level)', fontweight='bold')
    axes[0, 1].set_ylabel('Grade level')
    
    # BERTScore
    sns.barplot(ax=axes[1, 0], data=df_sorted, x='name_complexity', y='bertscore_f1',
                order=complexity_order, palette='Greens_d')
    axes[1, 0].set_title('BERTScore F1', fontweight='bold')
    axes[1, 0].set_ylabel('F1 score')
    axes[1, 0].set_ylim(0.7, 1)
    
    # Mutation Rate
    sns.barplot(ax=axes[1, 1], data=df_sorted, x='name_complexity', y='name_mutation_rate',
                order=complexity_order, palette='Reds_d')
    axes[1, 1].set_title('Name mutation rate', fontweight='bold')
    axes[1, 1].set_ylabel('Mutation rate')
    
    for ax in axes.flat:
        ax.set_xlabel('Name complexity')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_performance_by_complexity.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: bias_performance_by_complexity.png")
    plt.close()
    
    # 7. Provider Comparison
    print("📊 Generating: Provider bias comparison")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Bias metrics by API provider', fontsize=16, weight='bold')
    
    # Name Preservation by Provider
    sns.boxplot(ax=axes[0], data=df, x='api_provider', y='name_preservation_score', palette='Set2')
    axes[0].set_title('Name preservation', fontweight='bold')
    axes[0].set_ylabel('Preservation score')
    axes[0].set_ylim(0, 1)
    
    # BERTScore by provider
    sns.boxplot(ax=axes[1], data=df, x='api_provider', y='bertscore_f1', palette='Set2')
    axes[1].set_title('Semantic fidelity', fontweight='bold')
    axes[1].set_ylabel('BERTScore F1')
    axes[1].set_ylim(0.7, 1)
    
    # Mutation rate by provider
    sns.boxplot(ax=axes[2], data=df, x='api_provider', y='name_mutation_rate', palette='Set2')
    axes[2].set_title('Name mutation', fontweight='bold')
    axes[2].set_ylabel('Mutation rate')
    
    for ax in axes:
        ax.set_xlabel('API provider')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bias_provider_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: bias_provider_comparison.png")
    plt.close()

    print(f"Visualisations saved to {output_dir}")


if __name__ == "__main__":
    generate_bias_visuals()