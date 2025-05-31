import json
import re

def calculate_jaccard_and_update_models_count(file_path):
    """
    Parses a JSON file, calculates Jaccard similarity metrics,
    and updates the 'models_count' to 1000 for each entry.

    Args:
        file_path (str): The path to the JSON file.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return

    total_jaccard_similarity = 0
    total_jaccard_numerator = 0
    total_jaccard_denominator = 0
    valid_entries_count = 0

    for key, policy_data in data.items():
        # Update models_count
        if "models_count" in policy_data:
            policy_data["models_count"] = 1000

        # Calculate Jaccard similarity
        if "analysis" in policy_data and isinstance(policy_data["analysis"], str):
            analysis_text = policy_data["analysis"]
            
            numerator_match = re.search(r"jaccard_numerator\s*:\s*([\d.]+)", analysis_text)
            denominator_match = re.search(r"jaccard_denominator\s*:\s*([\d.]+)", analysis_text)

            if numerator_match and denominator_match:
                try:
                    numerator = float(numerator_match.group(1))
                    denominator = float(denominator_match.group(1))

                    if denominator != 0:
                        jaccard_similarity = numerator / denominator
                        total_jaccard_similarity += jaccard_similarity
                        total_jaccard_numerator += numerator
                        total_jaccard_denominator += denominator
                        valid_entries_count += 1
                        
                        # Optionally, store these back into the policy_data if needed
                        # policy_data["jaccard_similarity_calculated"] = jaccard_similarity
                        # policy_data["jaccard_numerator_calculated"] = numerator
                        # policy_data["jaccard_denominator_calculated"] = denominator
                    else:
                        print(f"Warning: Jaccard denominator is zero for policy {key}. Skipping.")
                except ValueError:
                    print(f"Warning: Could not convert Jaccard values to float for policy {key}. Skipping.")
            # else:
                # print(f"Warning: Jaccard numerator or denominator not found for policy {key}. Skipping.")


    if valid_entries_count > 0:
        avg_jaccard_similarity = total_jaccard_similarity / valid_entries_count
        avg_jaccard_numerator = total_jaccard_numerator / valid_entries_count
        avg_jaccard_denominator = total_jaccard_denominator / valid_entries_count

        print(f"Average Jaccard Similarity: {avg_jaccard_similarity}")
        print(f"Average Jaccard Numerator: {avg_jaccard_numerator}")
        print(f"Average Jaccard Denominator: {avg_jaccard_denominator}")
    else:
        print("No valid entries found to calculate averages.")

    # Write the updated data back to the file
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully updated {file_path}")
    except IOError:
        print(f"Error: Could not write updated JSON to {file_path}")

if __name__ == "__main__":
    # The user wants to process 'baseline/all_results.json'
    # The path should be relative to where the script is run,
    # or an absolute path.
    # Assuming the script will be in the workspace root.
    json_file_path = "baseline/all_results.json"
    calculate_jaccard_and_update_models_count(json_file_path) 