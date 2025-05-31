#!/usr/bin/env python3

import json
import shutil
from pathlib import Path
from typing import Dict, List

def merge_rerun_results():
    """
    Merge the re-run results from the temp directories back into the main checkpoint files.
    """
    
    # Models that had failed experiments
    models_with_reruns = ["claude-3.7-sonnet", "o4-mini"]
    failed_policies = {
        "claude-3.7-sonnet": ["4.json", "9.json"],
        "o4-mini": ["1.json", "12.json", "13.json", "15.json", "24.json", "26.json"]
    }
    
    for model in models_with_reruns:
        print(f"\n🔄 Processing {model}...")
        
        # Paths
        main_checkpoint = f"CPCA/experiment_results/experiment_checkpoint_{model}.json"
        rerun_checkpoint = f"CPCA/experiment_results/experiment_checkpoint_{model}.json"
        backup_checkpoint = f"CPCA/experiment_results/experiment_checkpoint_{model}_backup.json"
        
        # Create backup of original
        shutil.copy2(main_checkpoint, backup_checkpoint)
        print(f"   📦 Backup created: {backup_checkpoint}")
        
        # Load main checkpoint
        with open(main_checkpoint, 'r') as f:
            main_data = json.load(f)
        
        # Load re-run checkpoint (should contain only the re-run results)
        try:
            with open(rerun_checkpoint, 'r') as f:
                rerun_data = json.load(f)
        except FileNotFoundError:
            print(f"   ⚠️  No re-run checkpoint found for {model}")
            continue
        
        # Get the failed policy filenames for this model
        failed_policy_files = failed_policies[model]
        
        # Find and replace the failed experiments in main data
        updated_count = 0
        rerun_results = rerun_data.get('results', [])
        
        for rerun_result in rerun_results:
            rerun_policy_file = rerun_result.get('policy_file', '')
            
            # Find the corresponding experiment in main data and replace it
            for i, main_result in enumerate(main_data['results']):
                if main_result.get('policy_file') == rerun_policy_file:
                    print(f"   🔄 Updating {rerun_policy_file}")
                    main_data['results'][i] = rerun_result
                    updated_count += 1
                    break
        
        # Update the completed list to include the re-run experiments
        main_completed = set(main_data.get('completed', []))
        rerun_completed = set(rerun_data.get('completed', []))
        
        # Add the new completed experiments
        for policy_file in failed_policy_files:
            experiment_id = f"{model}::{policy_file}"
            main_completed.add(experiment_id)
        
        main_data['completed'] = list(main_completed)
        
        # Save the updated main checkpoint
        with open(main_checkpoint, 'w') as f:
            json.dump(main_data, f, indent=2)
        
        print(f"   ✅ Updated {updated_count} experiments for {model}")
        print(f"   💾 Saved to: {main_checkpoint}")
        
        # Verify the update worked
        verify_updates(model, failed_policy_files)

def verify_updates(model: str, failed_policy_files: List[str]):
    """Verify that the updates were successful by checking for JSON parsing errors"""
    
    checkpoint_file = f"CPCA/experiment_results/experiment_checkpoint_{model}.json"
    
    with open(checkpoint_file, 'r') as f:
        data = json.load(f)
    
    print(f"   🔍 Verifying updates for {model}:")
    
    still_failed = []
    fixed_count = 0
    
    for result in data.get('results', []):
        policy_file = result.get('policy_file', '')
        
        if policy_file in failed_policy_files:
            # Check if this experiment still has JSON parsing failure
            semantic_equiv = result.get('results', {}).get('semantic_equivalence', {})
            quacky_output = semantic_equiv.get('quacky_output', '')
            
            if 'Failed to parse reconstructed policy JSON' in quacky_output:
                still_failed.append(policy_file)
                print(f"      ❌ {policy_file}: Still has JSON parsing error")
            else:
                fixed_count += 1
                print(f"      ✅ {policy_file}: Fixed!")
    
    print(f"   📊 Summary: {fixed_count} fixed, {len(still_failed)} still failing")
    
    if still_failed:
        print(f"   ⚠️  Still failing: {still_failed}")
    else:
        print(f"   🎉 All JSON parsing errors fixed for {model}!")

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    
    print(f"\n🧹 Cleaning up temporary files...")
    
    # Remove temp policy directories
    temp_dirs = [
        "temp_failed_policies_claude-3.7-sonnet",
        "temp_failed_policies_o4-mini"
    ]
    
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
            print(f"   🗑️  Removed: {temp_dir}/")
    
    # Remove re-run scripts
    rerun_scripts = [
        "rerun_claude-3.7-sonnet_failed.sh",
        "rerun_o4-mini_failed.sh"
    ]
    
    for script in rerun_scripts:
        if Path(script).exists():
            Path(script).unlink()
            print(f"   🗑️  Removed: {script}")
    
    print(f"   ✅ Cleanup complete")

def main():
    print("🔧 Merging re-run results back into main checkpoint files...")
    
    # Merge the results
    merge_rerun_results()
    
    # Ask user if they want to clean up temp files
    print(f"\n❓ Would you like to clean up temporary files? (y/n)")
    response = input().strip().lower()
    
    if response in ['y', 'yes']:
        cleanup_temp_files()
    else:
        print(f"   📁 Temporary files kept for manual review")
    
    print(f"\n🎉 Merge process complete!")
    print(f"   📊 You can now run your analysis scripts to see the updated results")
    print(f"   📁 Backup files created with '_backup' suffix")

if __name__ == "__main__":
    main() 