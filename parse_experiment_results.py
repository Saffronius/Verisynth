import json
import statistics
from typing import List, Dict, Any
import sys

def parse_experiment_results(json_file_path: str) -> Dict[str, Any]:
    """
    Parse experiment results JSON file and extract specific metrics.
    
    Args:
        json_file_path: Path to the JSON file containing experiment results
        
    Returns:
        Dictionary containing extracted metrics and calculated averages
    """
    
    # Read and parse the JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Initialize lists to store extracted metrics
    explanation_consistencies = []
    request_comprehension_accuracies = []
    semantic_equivalences = []
    
    # Track counts for semantic equivalence
    semantic_equivalent_true_count = 0
    semantic_equivalent_false_count = 0
    
    # Extract metrics from each experiment in the results list
    results = data.get('results', [])
    
    for i, experiment in enumerate(results):
        try:
            # Extract explanation consistency
            explanation_consistency = experiment['results']['explanation_consistency']
            explanation_consistencies.append(explanation_consistency)
            
            # Extract request comprehension accuracy
            request_accuracy = experiment['results']['request_predictions']['accuracy']
            request_comprehension_accuracies.append(request_accuracy)
            
            # Extract semantic equivalence boolean
            semantic_equivalent = experiment['results']['semantic_equivalence']['semantic_equivalent']
            semantic_equivalences.append(semantic_equivalent)
            
            # Count semantic equivalence results
            if semantic_equivalent:
                semantic_equivalent_true_count += 1
            else:
                semantic_equivalent_false_count += 1
                
        except KeyError as e:
            print(f"Warning: Missing field {e} in experiment {i}")
            continue
        except Exception as e:
            print(f"Error processing experiment {i}: {e}")
            continue
    
    # Calculate averages
    avg_explanation_consistency = statistics.mean(explanation_consistencies) if explanation_consistencies else 0
    avg_request_comprehension_accuracy = statistics.mean(request_comprehension_accuracies) if request_comprehension_accuracies else 0
    
    # Prepare results dictionary
    results_summary = {
        'total_experiments': len(results),
        'successfully_processed': len(explanation_consistencies),
        'explanation_consistency': {
            'values': explanation_consistencies,
            'average': avg_explanation_consistency,
            'count': len(explanation_consistencies)
        },
        'request_comprehension_accuracy': {
            'values': request_comprehension_accuracies,
            'average': avg_request_comprehension_accuracy,
            'count': len(request_comprehension_accuracies)
        },
        'semantic_equivalence': {
            'values': semantic_equivalences,
            'true_count': semantic_equivalent_true_count,
            'false_count': semantic_equivalent_false_count,
            'total_count': len(semantic_equivalences),
            'true_percentage': (semantic_equivalent_true_count / len(semantic_equivalences) * 100) if semantic_equivalences else 0,
            'false_percentage': (semantic_equivalent_false_count / len(semantic_equivalences) * 100) if semantic_equivalences else 0
        }
    }
    
    return results_summary

def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the extracted results.
    
    Args:
        results: Dictionary containing the extracted metrics and calculated values
    """
    print("=" * 60)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)
    
    print(f"Total experiments in file: {results['total_experiments']}")
    print(f"Successfully processed: {results['successfully_processed']}")
    print()
    
    # Explanation Consistency
    print("EXPLANATION CONSISTENCY:")
    print(f"  Average: {results['explanation_consistency']['average']:.4f}")
    print(f"  Count: {results['explanation_consistency']['count']}")
    print()
    
    # Request Comprehension Accuracy
    print("REQUEST COMPREHENSION ACCURACY:")
    print(f"  Average: {results['request_comprehension_accuracy']['average']:.4f}")
    print(f"  Count: {results['request_comprehension_accuracy']['count']}")
    print()
    
    # Semantic Equivalence
    print("SEMANTIC EQUIVALENCE:")
    print(f"  Total evaluated: {results['semantic_equivalence']['total_count']}")
    print(f"  Equivalent (True): {results['semantic_equivalence']['true_count']} ({results['semantic_equivalence']['true_percentage']:.1f}%)")
    print(f"    -> 'Policy 1 and Policy 2 are equivalent'")
    print(f"  Not equivalent (False): {results['semantic_equivalence']['false_count']} ({results['semantic_equivalence']['false_percentage']:.1f}%)")
    print(f"    -> 'Policy 1 and Policy 2 do not subsume each other'")
    print()

def save_results_to_json(results: Dict[str, Any], output_file: str) -> None:
    """
    Save the extracted results to a JSON file.
    
    Args:
        results: Dictionary containing the extracted metrics
        output_file: Path to save the results JSON file
    """
    with open(output_file, 'w') as file:
        json.dump(results, file, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Check if a file path was provided as command line argument
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # Default to the original file if no argument provided
        json_file_path = "CPCA/experiment_results/experiment_checkpoint_grok-3.json"
    
    try:
        # Parse the experiment results
        results = parse_experiment_results(json_file_path)
        
        # Extract model name from filename for output
        model_name = json_file_path.split('/')[-1].replace('experiment_checkpoint_', '').replace('.json', '')
        
        # Print the summary with model name
        print(f"Processing: {json_file_path}")
        print(f"Model: {model_name}")
        print_results_summary(results)
        
        # Save results to a new JSON file with model name
        output_file = f"experiment_results_summary_{model_name}.json"
        save_results_to_json(results, output_file)
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        print("Please ensure the file path is correct.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}") 