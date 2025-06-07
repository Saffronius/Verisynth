import json
import re
from tqdm import tqdm
import decimal
import statistics

# Set higher precision for decimal calculations with large numbers
decimal.getcontext().prec = 100

def calculate_avg_similarity(results_file):
    """Calculate average Jaccard similarity from existing all_results.json"""
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    total_policies = len(results)
    processed_policies = 0
    excluded_perfect_matches = 0
    total_similarity = decimal.Decimal('0')
    
    # Store individual similarities for reporting
    similarities = {}
    perfect_matches = {}
    
    print(f"Processing {total_policies} policies...")
    
    # Process each policy
    for policy_num, policy_data in tqdm(results.items()):
        if not policy_data.get("success", False) or policy_data.get("analysis") == "TIMEOUT":
            print(f"Skipping policy {policy_num} (not successful or timeout)")
            continue
        
        analysis = policy_data.get("analysis", "")
        
        # Extract Jaccard metrics with the correct field names
        jaccard_num_match = re.search(r'jaccard_numerator\s+:\s+(\d+)', analysis)
        jaccard_denom_match = re.search(r'jaccard_denominator\s+:\s+(\d+)', analysis)
        
        if jaccard_num_match and jaccard_denom_match:
            # Use Decimal for handling very large numbers
            jaccard_num = decimal.Decimal(jaccard_num_match.group(1))
            jaccard_denom = decimal.Decimal(jaccard_denom_match.group(1))
            
            if jaccard_denom > 0:
                similarity = jaccard_num / jaccard_denom
                
                # Check if this is a perfect match (similarity = 1.0)
                if abs(similarity - decimal.Decimal('1.0')) < decimal.Decimal('0.0000001'):
                    perfect_matches[policy_num] = float(similarity)
                    excluded_perfect_matches += 1
                    print(f"Policy {policy_num} similarity: {float(similarity):.10f} (perfect match, excluded from average)")
                else:
                    similarities[policy_num] = float(similarity)
                    total_similarity += similarity
                    processed_policies += 1
                    print(f"Policy {policy_num} similarity: {float(similarity):.10f}")
            else:
                print(f"Policy {policy_num} has zero denominator")
        else:
            print(f"Could not find Jaccard metrics for policy {policy_num}")
    
    # Calculate average similarity (excluding perfect matches)
    avg_similarity = float(total_similarity / decimal.Decimal(processed_policies)) if processed_policies > 0 else 0
    
    return avg_similarity, processed_policies, total_policies, similarities, perfect_matches, excluded_perfect_matches, float(total_similarity)

if __name__ == "__main__":
    # Path to the all_results.json file
    results_file = "/mnt/d/Research/VeriSynth/Verifying-LLMAccessControl/Exp-4-Zelkova/results/all_results.json"
    
    # Calculate average similarity
    avg_similarity, processed, total, similarities, perfect_matches, excluded_count, sum_similarity = calculate_avg_similarity(results_file)
    
    print(f"\nSUMMARY:")
    print(f"Total policies in file: {total}")
    print(f"Successfully processed policies: {processed + excluded_count}")
    print(f"Perfect matches (similarity = 1.0) excluded: {excluded_count}")
    print(f"Policies included in average: {processed}")
    print(f"Sum of similarities (excluding perfect matches): {sum_similarity:.10f}")
    print(f"AVERAGE SIMILARITY (excluding perfect matches): {avg_similarity:.10f}")
    
    # Print individual similarities sorted by policy number
    print("\nIncluded similarities (excluding perfect matches):")
    for policy_num in sorted(similarities.keys(), key=int):
        print(f"Policy {policy_num}: {similarities[policy_num]:.10f}")
    
    # Print perfect matches
    if perfect_matches:
        print("\nExcluded perfect matches (similarity = 1.0):")
        for policy_num in sorted(perfect_matches.keys(), key=int):
            print(f"Policy {policy_num}: {perfect_matches[policy_num]:.10f}")
    
    # Calculate additional statistics
    if similarities:
        similarity_values = list(similarities.values())
        median = statistics.median(similarity_values)
        
        # Print summary with explicit calculation
        print("\n" + "="*50)
        print("FINAL RESULTS (EXCLUDING PERFECT MATCHES)")
        print("="*50)
        print(f"Total policies in file: {total}")
        print(f"Successfully processed policies: {processed + excluded_count}")
        print(f"Perfect matches (similarity = 1.0) excluded: {excluded_count}")
        print(f"Policies included in average: {processed}")
        print(f"Sum of similarities (excluding perfect matches): {sum_similarity:.10f}")
        print(f"AVERAGE SIMILARITY (excluding perfect matches): {sum_similarity:.10f} / {processed} = {avg_similarity:.10f}")
        print("="*50)