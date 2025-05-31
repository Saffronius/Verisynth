#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Set

def identify_failed_experiments() -> Dict[str, List[str]]:
    """
    Identify experiments that failed with 'Failed to parse reconstructed policy JSON'
    Returns a dictionary mapping model names to lists of policy files that need re-running
    """
    
    models = ["grok-3", "deepseek-chat", "claude-3.7-sonnet", "o4-mini"]
    failed_experiments = {}
    
    for model in models:
        checkpoint_file = f"CPCA/experiment_results/experiment_checkpoint_{model}.json"
        
        if not Path(checkpoint_file).exists():
            print(f"⚠️  Checkpoint file not found for {model}: {checkpoint_file}")
            continue
            
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            failed_policies = []
            results = data.get('results', [])
            
            print(f"\n📊 Analyzing {model}:")
            print(f"   Total experiments: {len(results)}")
            
            for result in results:
                policy_file = result.get('policy_file', 'unknown')
                
                # Check if this experiment had a JSON parsing failure
                semantic_equiv = result.get('results', {}).get('semantic_equivalence', {})
                quacky_output = semantic_equiv.get('quacky_output', '')
                
                if 'Failed to parse reconstructed policy JSON' in quacky_output:
                    failed_policies.append(policy_file)
                    print(f"   ❌ {policy_file}: {quacky_output[:80]}...")
            
            if failed_policies:
                failed_experiments[model] = failed_policies
                print(f"   🔄 Found {len(failed_policies)} policies to re-run for {model}")
            else:
                print(f"   ✅ No failed JSON parsing experiments found for {model}")
                
        except Exception as e:
            print(f"❌ Error reading checkpoint for {model}: {e}")
    
    return failed_experiments

def create_rerun_scripts(failed_experiments: Dict[str, List[str]]):
    """Create shell scripts to re-run only the failed experiments"""
    
    if not failed_experiments:
        print("\n🎉 No failed experiments found! All JSON parsing was successful.")
        return
    
    print(f"\n📝 Creating re-run scripts...")
    
    # Create a temporary policy directory for each model with only the failed policies
    base_policy_dir = "/home/ash/Desktop/VerifyingLLMGeneratedPolicies/Prev-Experiments/Verifying-LLMAccessControl/Dataset"
    
    for model, policies in failed_experiments.items():
        # Create temporary directory for this model's failed policies
        temp_dir = f"temp_failed_policies_{model}"
        Path(temp_dir).mkdir(exist_ok=True)
        
        # Copy failed policy files to temp directory
        for policy_file in policies:
            src_path = Path(base_policy_dir) / policy_file
            dst_path = Path(temp_dir) / policy_file
            
            if src_path.exists():
                # Copy the file content
                with open(src_path, 'r') as src:
                    policy_content = src.read()
                with open(dst_path, 'w') as dst:
                    dst.write(policy_content)
            else:
                print(f"⚠️  Source policy file not found: {src_path}")
        
        # Create the re-run script
        script_content = f"""#!/bin/bash

echo "🔄 Re-running failed experiments for {model}"
echo "Policies to re-run: {len(policies)}"

# Run CPCA for only the failed policies
python3 cpca.py \\
    --models {model} \\
    --policy-dir {temp_dir} \\
    --output-dir CPCA/experiment_results

echo "✅ Re-run complete for {model}"
echo "📁 Results saved to CPCA/experiment_results/experiment_checkpoint_{model}.json"
"""
        
        script_path = f"rerun_{model}_failed.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        os.chmod(script_path, 0o755)
        
        print(f"   📄 Created: {script_path}")
        print(f"   📁 Temp policies: {temp_dir}/ ({len(policies)} files)")

def merge_results(model: str, failed_policies: List[str]):
    """
    Merge the re-run results back into the main checkpoint file.
    This function will need to be called after re-running experiments.
    """
    
    main_checkpoint = f"CPCA/experiment_results/experiment_checkpoint_{model}.json"
    
    # Load main checkpoint
    with open(main_checkpoint, 'r') as f:
        main_data = json.load(f)
    
    # Update the results for the re-run policies
    # The new results should already be in the checkpoint file after re-running
    print(f"✅ Results for {model} should be updated in {main_checkpoint}")

def main():
    print("🔍 Identifying failed experiments with JSON parsing errors...")
    
    failed_experiments = identify_failed_experiments()
    
    if failed_experiments:
        print(f"\n📋 SUMMARY:")
        total_failed = 0
        for model, policies in failed_experiments.items():
            print(f"   {model}: {len(policies)} policies need re-running")
            total_failed += len(policies)
        
        print(f"\n🔢 Total experiments to re-run: {total_failed}")
        
        # Create re-run scripts
        create_rerun_scripts(failed_experiments)
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   1. Review the temp_failed_policies_* directories to verify the correct policies")
        print(f"   2. Run the generated rerun_*_failed.sh scripts:")
        
        for model in failed_experiments.keys():
            print(f"      ./rerun_{model}_failed.sh")
        
        print(f"   3. The results will be automatically merged into the existing checkpoint files")
        
        # Save the failed experiments list for reference
        with open('failed_experiments_list.json', 'w') as f:
            json.dump(failed_experiments, f, indent=2)
        print(f"   4. Failed experiments list saved to: failed_experiments_list.json")
        
    else:
        print(f"\n🎉 Great news! No failed experiments found.")
        print(f"   All models successfully parsed their reconstructed policy JSON.")

if __name__ == "__main__":
    main() 