#!/bin/bash

# Arrays of parameter values
temperatures=(0.1 0.5 0.9 1.3 1.7)
finetuned_values=(true false)

# Maximum number of parallel processes
max_parallel=5

# Function to run evaluation with given parameters
run_eval() {
    local temp=$1
    local use_finetuned=$2
    
    if [ "$use_finetuned" = true ]; then
        python3 prontoqa_eval.py --temp $temp --finetuned > /dev/null 2>&1
    else
        python3 prontoqa_eval.py --temp $temp > /dev/null 2>&1
    fi
}

# Array to store background process PIDs
declare -a pids

# Counter for running processes
count=0

# Iterate through all combinations
for temp in "${temperatures[@]}"; do
    for use_finetuned in "${finetuned_values[@]}"; do
        # If we've reached max parallel processes, wait for one to finish
        if [ $count -ge $max_parallel ]; then
            # Wait for any process to finish
            wait -n
            # Decrease counter
            ((count--))
        fi
        
        # Run the evaluation in background
        echo "Starting evaluation with temp=$temp, finetuned=$([ "$use_finetuned" = true ] && echo "yes" || echo "no")"
        run_eval $temp $use_finetuned &
        
        # Store the PID
        pids+=($!)
        
        # Increment counter
        ((count++))
    done
done

# Wait for remaining processes to finish
wait

echo "All evaluations completed!"