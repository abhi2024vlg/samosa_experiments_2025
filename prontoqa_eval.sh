#!/bin/bash

# Arrays of parameter values
temperatures=(0.1 0.3 0.5)

# Maximum number of parallel processes
max_parallel=5

# Function to run evaluation with given parameters
run_eval() {
    local temp=$1
    local p_val=$2
    python3 script.py --temp $temp --p $p_val
}

# Array to store background process PIDs
declare -a pids

# Counter for running processes
count=0

# Iterate through all combinations
for temp in "${temperatures[@]}"; do
    for p_val in "${p_values[@]}"; do
        # If we've reached max parallel processes, wait for one to finish
        if [ $count -ge $max_parallel ]; then
            # Wait for any process to finish
            wait -n
            # Decrease counter
            ((count--))
        fi
        
        # Run the evaluation in background
        echo "Starting evaluation with temp=$temp, p=$p_val"
        run_eval $temp $p_val &
        
        # Store the PID
        pids+=($!)
        echo "Current PIDs: ${pids[@]}"
        
        # Increment counter
        ((count++))
    done
done

# Wait for remaining processes to finish
wait

echo "All evaluations completed!"