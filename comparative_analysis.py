import json
import pandas as pd
from typing import Dict, List

def load_results(model_name: str) -> Dict:
    """Load results for a specific model."""
    filename = f"experiment_results_summary_{model_name}.json"
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: Results file for {model_name} not found.")
        return None

def create_comparative_table():
    """Create a comparative analysis table for all models."""
    
    models = ["grok-3", "deepseek-chat", "claude-3.7-sonnet", "o4-mini"]
    
    # Collect data for all models
    comparison_data = []
    
    for model in models:
        results = load_results(model)
        if results:
            comparison_data.append({
                'Model': model,
                'Explanation Consistency (Avg)': round(results['explanation_consistency']['average'], 4),
                'Request Comprehension Accuracy (Avg)': round(results['request_comprehension_accuracy']['average'], 4),
                'Semantic Equivalent (True)': results['semantic_equivalence']['true_count'],
                'Semantic Equivalent (True %)': round(results['semantic_equivalence']['true_percentage'], 1),
                'Semantic Not Equivalent (False)': results['semantic_equivalence']['false_count'],
                'Semantic Not Equivalent (False %)': round(results['semantic_equivalence']['false_percentage'], 1),
                'Total Experiments': results['total_experiments']
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Print formatted table
    print("=" * 120)
    print("COMPARATIVE ANALYSIS: ALL MODELS")
    print("=" * 120)
    print()
    
    # Print the table with nice formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print(df.to_string(index=False))
    
    print("\n" + "=" * 120)
    print("RANKING BY METRICS:")
    print("=" * 120)
    
    # Ranking by Explanation Consistency
    print("\n📊 EXPLANATION CONSISTENCY (Highest to Lowest):")
    explanation_ranking = df.sort_values('Explanation Consistency (Avg)', ascending=False)
    for i, row in explanation_ranking.iterrows():
        print(f"  {row.name + 1}. {row['Model']}: {row['Explanation Consistency (Avg)']}")
    
    # Ranking by Request Comprehension Accuracy
    print("\n🎯 REQUEST COMPREHENSION ACCURACY (Highest to Lowest):")
    accuracy_ranking = df.sort_values('Request Comprehension Accuracy (Avg)', ascending=False)
    for i, row in accuracy_ranking.iterrows():
        print(f"  {row.name + 1}. {row['Model']}: {row['Request Comprehension Accuracy (Avg)']}")
    
    # Ranking by Semantic Equivalence True %
    print("\n🔗 SEMANTIC EQUIVALENCE TRUE % (Highest to Lowest):")
    semantic_ranking = df.sort_values('Semantic Equivalent (True %)', ascending=False)
    for i, row in semantic_ranking.iterrows():
        print(f"  {row.name + 1}. {row['Model']}: {row['Semantic Equivalent (True %)']}%")
    
    print("\n" + "=" * 120)
    print("SUMMARY INSIGHTS:")
    print("=" * 120)
    
    # Best performing model in each category
    best_explanation = df.loc[df['Explanation Consistency (Avg)'].idxmax()]
    best_accuracy = df.loc[df['Request Comprehension Accuracy (Avg)'].idxmax()]
    best_semantic = df.loc[df['Semantic Equivalent (True %)'].idxmax()]
    
    print(f"\n🏆 Best Explanation Consistency: {best_explanation['Model']} ({best_explanation['Explanation Consistency (Avg)']})")
    print(f"🏆 Best Request Comprehension: {best_accuracy['Model']} ({best_accuracy['Request Comprehension Accuracy (Avg)']})")
    print(f"🏆 Highest Semantic Equivalence: {best_semantic['Model']} ({best_semantic['Semantic Equivalent (True %)']}%)")
    
    # Save to CSV for further analysis
    df.to_csv('model_comparison.csv', index=False)
    print(f"\n💾 Comparative data saved to: model_comparison.csv")
    
    return df

if __name__ == "__main__":
    try:
        create_comparative_table()
    except ImportError:
        print("pandas library is required for this script. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "pandas"])
        import pandas as pd
        create_comparative_table()
    except Exception as e:
        print(f"Error: {e}") 