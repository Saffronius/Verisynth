#!/bin/bash

echo "🔄 Re-running failed experiments for o4-mini"
echo "Policies to re-run: 6"

# Activate virtual environment and change to the directory containing cpca.py
source ../venv/bin/activate
cd CPCA

# Run CPCA for only the failed policies with --from-scratch to force re-run
python3 cpca.py \
    --models o4-mini \
    --policy-dir ../temp_failed_policies_o4-mini \
    --output-dir experiment_results \
    --from-scratch

echo "✅ Re-run complete for o4-mini"
echo "📁 Results saved to CPCA/experiment_results/experiment_checkpoint_o4-mini.json"
