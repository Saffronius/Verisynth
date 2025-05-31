#!/bin/bash

echo "🔄 Re-running failed experiments for claude-3.7-sonnet"
echo "Policies to re-run: 2"

# Activate virtual environment and change to the directory containing cpca.py
source ../venv/bin/activate
cd CPCA

# Run CPCA for only the failed policies with --from-scratch to force re-run
python3 cpca.py \
    --models claude-3.7-sonnet \
    --policy-dir ../temp_failed_policies_claude-3.7-sonnet \
    --output-dir experiment_results \
    --from-scratch

echo "✅ Re-run complete for claude-3.7-sonnet"
echo "📁 Results saved to CPCA/experiment_results/experiment_checkpoint_claude-3.7-sonnet.json"
