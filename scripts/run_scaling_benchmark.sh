#!/bin/bash

# Comprehensive scaling benchmark for argmax operations
# Formula: N = 100 * 2^K, M = N/10
# Tests different implementations up to their practical limits

set -e

# Configuration
PROJECT_ROOT="/home/zhangyih/gpu_argmax"
RESULTS_DIR="scaling_results_$(date +%Y%m%d_%H%M%S)"
C_BINARY="$PROJECT_ROOT/reference_implementation/argmax_benchmark"

# Create results directory
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR"

# Initialize CSV files
echo "Implementation,Device,K,N,M,Time_ms,Std_ms,Status,Notes" > scaling_results.csv

# Logging function
log() {
    echo "[$(date '+%H:%M:%S')] $1" | tee -a benchmark.log
}

# Calculate N and M from K
calc_params() {
    local k=$1
    local n=$((100 * (2**k)))
    local m=$((n / 10))
    echo "$n $m"
}

# Test C Reference implementation
test_c_reference() {
    local k=$1
    local n=$2
    local m=$3
    
    log "Testing C Reference: K=$k, N=$n, M=$m"
    
    if [ ! -x "$C_BINARY" ]; then
        log "ERROR: C binary not found or not executable at $C_BINARY"
        echo "C Reference,CPU,$k,$n,$m,0,0,ERROR,Binary not found" >> scaling_results.csv
        return
    fi
    
    # Run without timeout
    if "$C_BINARY" "$n" "$m" 5 > "c_output_k${k}.txt" 2>&1; then
        # Parse output for timing
        local avg_time=$(grep "Average time:" "c_output_k${k}.txt" | awk '{print $3}')
        if [ -n "$avg_time" ]; then
            echo "C Reference,CPU,$k,$n,$m,$avg_time,0,SUCCESS," >> scaling_results.csv
            log "  ✓ C Reference completed: ${avg_time}ms"
        else
            echo "C Reference,CPU,$k,$n,$m,0,0,ERROR,Could not parse output" >> scaling_results.csv
            log "  ✗ C Reference failed to parse output"
        fi
    else
        echo "C Reference,CPU,$k,$n,$m,0,0,ERROR,Execution failed" >> scaling_results.csv
        log "  ✗ C Reference execution failed"
    fi
}

# Test PyTorch implementation
test_pytorch() {
    local device=$1
    local k=$2
    local n=$3
    local m=$4
    
    log "Testing PyTorch $device: K=$k, N=$n, M=$m"
    
    cd "$PROJECT_ROOT"
    
    if python run.py benchmark --n "$n" --m "$m" --device "$device" > "$RESULTS_DIR/pytorch_${device}_k${k}.txt" 2>&1; then
        # Parse output for full method timing 
        local core_time=$(grep "Full method:" "$RESULTS_DIR/pytorch_${device}_k${k}.txt" | awk -F'±' '{print $1}' | awk '{print $NF}')
        local core_std=$(grep "Full method:" "$RESULTS_DIR/pytorch_${device}_k${k}.txt" | awk -F'±' '{print $2}' | awk '{print $1}')
        
        if [ -n "$core_time" ] && [ -n "$core_std" ]; then
            echo "PyTorch (Full),${device^^},$k,$n,$m,$core_time,$core_std,SUCCESS," >> "$RESULTS_DIR/scaling_results.csv"
            log "  ✓ PyTorch $device completed: ${core_time}±${core_std}ms"
        else
            echo "PyTorch (Full),${device^^},$k,$n,$m,0,0,ERROR,Could not parse output" >> "$RESULTS_DIR/scaling_results.csv"
            log "  ✗ PyTorch $device failed to parse output"
        fi
    else
        echo "PyTorch (Full),${device^^},$k,$n,$m,0,0,ERROR,Execution failed" >> "$RESULTS_DIR/scaling_results.csv"
        log "  ✗ PyTorch $device execution failed"
    fi
    
    cd "$RESULTS_DIR"
}

# Main benchmark execution
main() {
    log "=== Starting Comprehensive Scaling Benchmark ==="
    log "Results directory: $RESULTS_DIR"
    
    # System information
    log "System Information:"
    log "  CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
    log "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
    if command -v nvidia-smi &> /dev/null; then
        log "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    else
        log "  GPU: Not available"
    fi
    
    # Test C Reference (K=0 to K=7, max N=12,800)
    log "\n=== Testing C Reference Implementation ==="
    for k in {0..7}; do
        read n m <<< $(calc_params $k)
        if [ $n -le 20000 ]; then
            test_c_reference $k $n $m
        else
            log "Skipping C Reference K=$k (N=$n > 20,000 limit)"
            break
        fi
    done
    
    # Test PyTorch CPU (K=0 to K=10, max N=102,400)  
    log "\n=== Testing PyTorch CPU Implementation ==="
    for k in {0..10}; do
        read n m <<< $(calc_params $k)
        if [ $n -le 200000 ]; then
            test_pytorch "cpu" $k $n $m
        else
            log "Skipping PyTorch CPU K=$k (N=$n > 200,000 limit)"
            break
        fi
    done
    
    # Test PyTorch GPU (K=0 to K=14, max N=1,638,400)
    if command -v nvidia-smi &> /dev/null; then
        log "\n=== Testing PyTorch GPU Implementation ==="
        for k in {0..14}; do
            read n m <<< $(calc_params $k)
            if [ $n -le 2000000 ]; then
                test_pytorch "cuda" $k $n $m
            else
                log "Skipping PyTorch GPU K=$k (N=$n > 2,000,000 limit)"
                break
            fi
        done
    else
        log "\n=== Skipping PyTorch GPU (NVIDIA GPU not available) ==="
    fi
    
    log "\n=== Benchmark Complete ==="
    log "Results saved in: $RESULTS_DIR/"
    log "  - scaling_results.csv: Main results table"
    log "  - benchmark.log: Detailed execution log"
    log "  - *_output_k*.txt: Individual test outputs"
    
    # Generate summary
    log "\n=== Results Summary ==="
    if [ -f scaling_results.csv ]; then
        log "Total tests run: $(($(wc -l < scaling_results.csv) - 1))"
        log "Successful tests: $(grep -c SUCCESS scaling_results.csv)"
        log "Failed/timeout tests: $(grep -c -E "(ERROR|TIMEOUT)" scaling_results.csv)"
    fi
    
    echo ""
    echo "To analyze results, you can:"
    echo "  1. View CSV: cat $RESULTS_DIR/scaling_results.csv"
    echo "  2. Import to Python/Excel for plotting"
    echo "  3. Check individual outputs in $RESULTS_DIR/"
}

# Execute main function
main "$@"