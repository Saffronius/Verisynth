import json

def load_results(model_name: str):
    """Load results for a specific model."""
    filename = f"experiment_results_summary_{model_name}.json"
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Warning: Results file for {model_name} not found.")
        return None

def print_comparison():
    """Print a formatted comparison of all models."""
    
    models = ["grok-3", "deepseek-chat", "claude-3.7-sonnet", "o4-mini"]
    results_data = {}
    
    # Load all results
    for model in models:
        results = load_results(model)
        if results:
            results_data[model] = results
    
    print("=" * 100)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("=" * 100)
    print()
    
    # Table header
    print(f"{'Model':<20} {'Expl. Consistency':<18} {'Request Accuracy':<16} {'Semantic True':<14} {'Semantic False':<15}")
    print("-" * 100)
    
    # Table data
    for model, results in results_data.items():
        explanation_avg = results['explanation_consistency']['average']
        accuracy_avg = results['request_comprehension_accuracy']['average']
        semantic_true = results['semantic_equivalence']['true_count']
        semantic_false = results['semantic_equivalence']['false_count']
        true_percent = results['semantic_equivalence']['true_percentage']
        false_percent = results['semantic_equivalence']['false_percentage']
        
        print(f"{model:<20} {explanation_avg:<18.4f} {accuracy_avg:<16.4f} {semantic_true:>2} ({true_percent:>5.1f}%) {semantic_false:>3} ({false_percent:>5.1f}%)")
    
    print()
    print("=" * 100)
    print("DETAILED METRICS BREAKDOWN")
    print("=" * 100)
    
    for model, results in results_data.items():
        print(f"\n📊 {model.upper()}:")
        print(f"   Explanation Consistency: {results['explanation_consistency']['average']:.4f}")
        print(f"   Request Comprehension:   {results['request_comprehension_accuracy']['average']:.4f}")
        print(f"   Semantic Equivalence:    {results['semantic_equivalence']['true_count']}/41 ({results['semantic_equivalence']['true_percentage']:.1f}%)")
        print(f"   Non-Equivalence:         {results['semantic_equivalence']['false_count']}/41 ({results['semantic_equivalence']['false_percentage']:.1f}%)")
    
    print("\n" + "=" * 100)
    print("RANKINGS")
    print("=" * 100)
    
    # Sort by different metrics
    models_with_scores = []
    for model, results in results_data.items():
        models_with_scores.append({
            'model': model,
            'explanation': results['explanation_consistency']['average'],
            'accuracy': results['request_comprehension_accuracy']['average'],
            'semantic_true_pct': results['semantic_equivalence']['true_percentage']
        })
    
    # Explanation Consistency ranking
    print("\n🏆 EXPLANATION CONSISTENCY (Best to Worst):")
    sorted_by_explanation = sorted(models_with_scores, key=lambda x: x['explanation'], reverse=True)
    for i, item in enumerate(sorted_by_explanation, 1):
        print(f"   {i}. {item['model']}: {item['explanation']:.4f}")
    
    # Request Comprehension ranking
    print("\n🎯 REQUEST COMPREHENSION ACCURACY (Best to Worst):")
    sorted_by_accuracy = sorted(models_with_scores, key=lambda x: x['accuracy'], reverse=True)
    for i, item in enumerate(sorted_by_accuracy, 1):
        print(f"   {i}. {item['model']}: {item['accuracy']:.4f}")
    
    # Semantic Equivalence ranking
    print("\n🔗 SEMANTIC EQUIVALENCE % (Best to Worst):")
    sorted_by_semantic = sorted(models_with_scores, key=lambda x: x['semantic_true_pct'], reverse=True)
    for i, item in enumerate(sorted_by_semantic, 1):
        print(f"   {i}. {item['model']}: {item['semantic_true_pct']:.1f}%")
    
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    
    best_explanation = max(models_with_scores, key=lambda x: x['explanation'])
    best_accuracy = max(models_with_scores, key=lambda x: x['accuracy'])
    best_semantic = max(models_with_scores, key=lambda x: x['semantic_true_pct'])
    
    print(f"\n✨ Overall Best Performer:")
    print(f"   • Explanation Consistency: {best_explanation['model']} ({best_explanation['explanation']:.4f})")
    print(f"   • Request Comprehension: {best_accuracy['model']} ({best_accuracy['accuracy']:.4f})")
    print(f"   • Semantic Equivalence: {best_semantic['model']} ({best_semantic['semantic_true_pct']:.1f}%)")
    
    # Overall performance calculation (simple average of normalized scores)
    print(f"\n📈 Overall Performance Score (normalized average):")
    
    # Normalize scores to 0-1 range
    max_explanation = max(item['explanation'] for item in models_with_scores)
    max_accuracy = max(item['accuracy'] for item in models_with_scores)
    max_semantic = max(item['semantic_true_pct'] for item in models_with_scores)
    
    overall_scores = []
    for item in models_with_scores:
        normalized_explanation = item['explanation'] / max_explanation
        normalized_accuracy = item['accuracy'] / max_accuracy
        normalized_semantic = item['semantic_true_pct'] / max_semantic
        overall_score = (normalized_explanation + normalized_accuracy + normalized_semantic) / 3
        overall_scores.append({'model': item['model'], 'score': overall_score})
    
    overall_scores.sort(key=lambda x: x['score'], reverse=True)
    for i, item in enumerate(overall_scores, 1):
        print(f"   {i}. {item['model']}: {item['score']:.4f}")

if __name__ == "__main__":
    print_comparison() 