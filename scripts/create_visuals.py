# create_visuals.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path # Path object
import numpy as np # Path object

# Loads eval results and generates benchmark visualisations for AT2 project poster
def create_poster_visuals(input_filename='advanced_evaluation_results.csv'):

    # Define project root relative to script location
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Build paths relative to project root
    filepath = PROJECT_ROOT / "outputs" / input_filename
    output_dir = PROJECT_ROOT / "outputs"

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found")
        return

    # Set pretty plot style
    sns.set_theme(style="whitegrid", palette="viridis")

    # Visualisation option 1: Before/after readability improvement paired box plot   
    # First, restructure data for plot prep
    readability_data = pd.melt(df, 
                               id_vars=['citation'], 
                               value_vars=['original_readability_grade', 'summary_readability_grade'],
                               var_name='text_type', 
                               value_name='readability_grade')
    
    # define more interesting variable names
    readability_data['text_type'] = readability_data['text_type'].map({
        'original_readability_grade': 'Readability before (original text)',
        'summary_readability_grade': 'Readability after (generated summary)'
    })

    plt.figure(figsize=(10, 8))
    sns.boxplot(x='text_type', y='readability_grade', data=readability_data, palette=["#ffde59", "#2d6aa2"])
    sns.stripplot(x='text_type', y='readability_grade', data=readability_data, color=".25", s=4, alpha=0.1)
    
    plt.title('LegalEase NSW dramatically reduces text complexity', fontsize=18, weight='bold')
    plt.ylabel('Flesch-Kincaid grade level', fontsize=14)
    plt.xlabel('') # Hide x-axis label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Check output directory exists before saving, learned this the hard way
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename_1 = output_dir / 'readability_improvement_visual.png'
    plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_1}")

    # Visualisation option 2: Paired histogram/density plot to assess what looks better
    plt.figure(figsize=(12, 7))
    sns.kdeplot(data=df, x='original_readability_grade', fill=True, label='Before (original text)')
    sns.kdeplot(data=df, x='summary_readability_grade', fill=True, label='After (generated summary)')
    
    plt.title('Dramatic shift in readability score distribution', fontsize=18, weight='bold')
    plt.xlabel('Flesch-Kincaid grade level', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    output_filename_2 = output_dir / 'density_plot_visual.png'
    plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_2}")

    # Visualisation option 3: Violin plot to assess what looks better
    plt.figure(figsize=(10, 8))
    sns.violinplot(x='text_type', y='readability_grade', data=readability_data, inner='quartile', palette='viridis')

    plt.title('Model summarises legislation into consistent range', fontsize=18, weight='bold')
    plt.ylabel('Flesch-Kincaid grade level', fontsize=14)
    plt.xlabel('') # Hide x-axis label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    output_filename_3 = output_dir / 'violin_plot_visual.png'
    plt.savefig(output_filename_3, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_3}")

    # Visualisation option 4: Slope chart to assess what looks better
    sample_df = df.sample(n=50, random_state=42) if len(df) > 50 else df
    
    plt.figure(figsize=(10, 8))
    for index, row in sample_df.iterrows():
        plt.plot(['Before', 'After'], [row['original_readability_grade'], row['summary_readability_grade']], marker='o', linestyle='-', color='gray', alpha=0.5)

    plt.plot(['Before'] * len(sample_df), sample_df['original_readability_grade'], 'o', color='C0', markersize=8, label='Original')
    plt.plot(['After'] * len(sample_df), sample_df['summary_readability_grade'], 'o', color='C1', markersize=8, label='Summary')

    plt.title('Readability improvement for individual documents', fontsize=18, weight='bold')
    plt.ylabel('Flesch-Kincaid grade level', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x')

    output_filename_4 = output_dir / 'slope_chart_visual.png'
    plt.savefig(output_filename_4, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_4}")

    # Visualisation option 5: Bar chart of averages to assess what looks better
    plt.figure(figsize=(8, 7))
    sns.barplot(x='text_type', y='readability_grade', data=readability_data, palette='viridis', ci='sd')

    plt.title('Average readability drops significantly', fontsize=18, weight='bold')
    plt.ylabel('Average Flesch-Kincaid grade level', fontsize=14)
    plt.xlabel('') # Hide x-axis label
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    output_filename_5 = output_dir / 'bar_chart_visual.png'
    plt.savefig(output_filename_5, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_5}")

    # Visualisation option 6: Before vs after scatter plot
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='original_readability_grade', y='summary_readability_grade', alpha=0.7, s=80)
    
    max_val = df['original_readability_grade'].max()
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', lw=2, label='No change line')

    plt.title('All summaries are simpler than originals', fontsize=18, weight='bold')
    plt.xlabel('Original readability (grade level)', fontsize=14)
    plt.ylabel('Summary readability (grade level)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.axis('equal')

    output_filename_6 = output_dir / 'before_after_scatter_visual.png'
    plt.savefig(output_filename_6, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_6}")

    # Visualisation 7: How readability improved (Flesch-Kincaid vs ROUGE) scatter plot
    plt.figure(figsize=(11, 7))
    sns.regplot(x='rougeL_fmeasure', y='readability_improvement', data=df,
                scatter_kws={'alpha':0.5, 's':20, 'color': '#ffde59'},
                line_kws={"color": "#2d6aa2", "lw":1.5})

    plt.title('Simplification vs factual overlap', fontsize=18, weight='bold')
    plt.xlabel('ROUGE-L F-measure (factual overlap)', fontsize=14)
    plt.ylabel('Readability improvement (grade levels)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Format x-axis as percentage
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0%}'.format))
    
    output_filename_7 = output_dir / 'readability_vs_rouge_visual.png'
    plt.savefig(output_filename_7, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_filename_7}")
    
if __name__ == "__main__":
    create_poster_visuals()