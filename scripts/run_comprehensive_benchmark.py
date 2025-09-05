#!/usr/bin/env python3
"""
Comprehensive benchmark runner for argmax operations.
Tests C reference, PyTorch CPU, and PyTorch CUDA implementations.
"""
import os
import sys
import time
import json
import subprocess
import pandas as pd
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(project_root)

from scripts.benchmark_argmax import run_benchmark

def get_system_info():
    """Get system information for the benchmark report."""
    try:
        # CPU info
        with open('/proc/cpuinfo', 'r') as f:
            cpu_info = [line.strip() for line in f if line.startswith('model name')]
            cpu_name = cpu_info[0].split(':')[1].strip() if cpu_info else "Unknown"
        
        # Memory info
        with open('/proc/meminfo', 'r') as f:
            mem_lines = f.readlines()
            mem_total = [line for line in mem_lines if line.startswith('MemTotal:')][0]
            mem_gb = int(mem_total.split()[1]) / 1024 / 1024
        
        # GPU info (if available)
        gpu_info = "Not available"
        try:
            nvidia_result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                         capture_output=True, text=True, timeout=5)
            if nvidia_result.returncode == 0:
                gpu_info = nvidia_result.stdout.strip()
        except:
            pass
        
        return {
            'cpu': cpu_name,
            'memory_gb': f"{mem_gb:.1f} GB",
            'gpu': gpu_info,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {'error': str(e), 'timestamp': datetime.now().isoformat()}

def run_c_benchmark(N, M, num_replications=10):
    """Run the C reference benchmark."""
    try:
        c_binary = os.path.join(project_root, 'reference_implementation', 'argmax_benchmark')
        if not os.path.exists(c_binary):
            print(f"C binary not found at {c_binary}. Please run 'make' in reference_implementation/")
            return None
        
        cmd = [c_binary, str(N), str(M), str(num_replications)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"C benchmark failed: {result.stderr}")
            return None
        
        # Parse the output to extract timing
        lines = result.stdout.strip().split('\n')
        avg_time_line = [line for line in lines if 'Average time:' in line]
        if not avg_time_line:
            print(f"Could not parse C benchmark output:\n{result.stdout}")
            return None
        
        avg_time_str = avg_time_line[0].split('Average time:')[1].strip().split()[0]
        avg_time_ms = float(avg_time_str)
        
        return {
            'implementation': 'C Reference',
            'device': 'CPU',
            'N': N,
            'M': M,
            'avg_time_ms': avg_time_ms,
            'std_time_ms': 0.0,  # C implementation doesn't calculate std
            'full_output': result.stdout
        }
    
    except subprocess.TimeoutExpired:
        print(f"C benchmark timed out for N={N}, M={M}")
        return None
    except Exception as e:
        print(f"Error running C benchmark: {e}")
        return None

def run_pytorch_benchmark(N, M, device):
    """Run the PyTorch benchmark."""
    try:
        print(f"Running PyTorch benchmark (N={N}, M={M}, device={device})...")
        result = run_benchmark(N, M, device)
        
        if result is None:
            return None
        
        return {
            'implementation': 'PyTorch',
            'device': device.upper(),
            'N': N,
            'M': M,
            'full_method_avg_ms': result['full_method_avg_ms'],
            'full_method_std_ms': result['full_method_std_ms'],
            'core_argmax_avg_ms': result['core_argmax_avg_ms'],
            'core_argmax_std_ms': result['core_argmax_std_ms'],
            'cut_coeff_avg_ms': result['cut_coeff_avg_ms'],
            'cut_coeff_std_ms': result['cut_coeff_std_ms']
        }
    
    except Exception as e:
        print(f"Error running PyTorch benchmark ({device}): {e}")
        return None

def main():
    """Run comprehensive benchmarks."""
    print("=== Comprehensive Argmax Benchmark Suite ===")
    print(f"Starting at {datetime.now()}")
    
    # Get system info
    system_info = get_system_info()
    print(f"System: {system_info.get('cpu', 'Unknown CPU')}")
    print(f"Memory: {system_info.get('memory_gb', 'Unknown')}")
    print(f"GPU: {system_info.get('gpu', 'Not available')}")
    
    # Test configurations
    test_configs = [
        (1000, 100),   # Small test
        (5000, 500),   # Medium test
        (10000, 1000), # Large test
    ]
    
    results = []
    
    # Test each configuration
    for N, M in test_configs:
        print(f"\n--- Testing N={N}, M={M} ---")
        
        # C Reference benchmark
        print("Running C reference...")
        c_result = run_c_benchmark(N, M)
        if c_result:
            results.append(c_result)
            print(f"C Reference: {c_result['avg_time_ms']:.2f} ms")
        
        # PyTorch CPU benchmark
        print("Running PyTorch CPU...")
        pytorch_cpu_result = run_pytorch_benchmark(N, M, 'cpu')
        if pytorch_cpu_result:
            results.append(pytorch_cpu_result)
            print(f"PyTorch CPU (full): {pytorch_cpu_result['full_method_avg_ms']:.2f} ms")
            print(f"PyTorch CPU (core): {pytorch_cpu_result['core_argmax_avg_ms']:.2f} ms")
        
        # PyTorch CUDA benchmark (if available)
        if system_info.get('gpu') != 'Not available':
            print("Running PyTorch CUDA...")
            pytorch_cuda_result = run_pytorch_benchmark(N, M, 'cuda')
            if pytorch_cuda_result:
                results.append(pytorch_cuda_result)
                print(f"PyTorch CUDA (full): {pytorch_cuda_result['full_method_avg_ms']:.2f} ms")
                print(f"PyTorch CUDA (core): {pytorch_cuda_result['core_argmax_avg_ms']:.2f} ms")
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    
    output_data = {
        'system_info': system_info,
        'test_configs': test_configs,
        'results': results,
        'timestamp': timestamp
    }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nRaw results saved to {results_file}")
    
    # Generate report
    generate_report(output_data, f"benchmark_report_{timestamp}")
    
    print("\n=== Benchmark Suite Complete ===")

def generate_report(data, report_prefix):
    """Generate markdown report and CSV files."""
    # Create comparison table for core argmax operations
    comparison_data = []
    
    for result in data['results']:
        if result['implementation'] == 'C Reference':
            comparison_data.append({
                'Implementation': 'C Reference',
                'Device': 'CPU',
                'N': result['N'],
                'M': result['M'],
                'Time (ms)': result['avg_time_ms'],
                'Std (ms)': 0.0
            })
        else:  # PyTorch
            comparison_data.append({
                'Implementation': f"PyTorch (Core)",
                'Device': result['device'],
                'N': result['N'],
                'M': result['M'],
                'Time (ms)': result['core_argmax_avg_ms'],
                'Std (ms)': result['core_argmax_std_ms']
            })
    
    # Create DataFrame and save CSV
    df = pd.DataFrame(comparison_data)
    csv_file = f"{report_prefix}_comparison.csv"
    df.to_csv(csv_file, index=False)
    
    # Generate markdown report
    md_file = f"{report_prefix}.md"
    with open(md_file, 'w') as f:
        f.write("# Argmax Operation Benchmark Report\n\n")
        f.write(f"Generated: {data['system_info']['timestamp']}\n\n")
        
        # System information
        f.write("## System Information\n\n")
        f.write(f"- **CPU**: {data['system_info'].get('cpu', 'Unknown')}\n")
        f.write(f"- **Memory**: {data['system_info'].get('memory_gb', 'Unknown')}\n")
        f.write(f"- **GPU**: {data['system_info'].get('gpu', 'Not available')}\n\n")
        
        # Problem dimensions
        f.write("## Problem Dimensions\n\n")
        f.write("Based on SSN problem instance:\n")
        f.write("- **M2 (Stage 2 rows)**: 175\n")
        f.write("- **K (Stochastic elements)**: 86\n")
        f.write("- **X_DIM (Stage 1 vars)**: 89\n\n")
        
        # Results table
        f.write("## Core Argmax Performance Comparison\n\n")
        f.write("This table compares the core argmax operation times for fair comparison:\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
        
        # Detailed PyTorch results
        pytorch_results = [r for r in data['results'] if r['implementation'] == 'PyTorch']
        if pytorch_results:
            f.write("## Detailed PyTorch Results\n\n")
            f.write("| Device | N | M | Full Method (ms) | Core Argmax (ms) | Cut Coeff (ms) |\n")
            f.write("|--------|---|---|------------------|------------------|----------------|\n")
            
            for result in pytorch_results:
                f.write(f"| {result['device']} | {result['N']} | {result['M']} | ")
                f.write(f"{result['full_method_avg_ms']:.2f} ± {result['full_method_std_ms']:.2f} | ")
                f.write(f"{result['core_argmax_avg_ms']:.2f} ± {result['core_argmax_std_ms']:.2f} | ")
                f.write(f"{result['cut_coeff_avg_ms']:.2f} ± {result['cut_coeff_std_ms']:.2f} |\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        f.write("### Implementation Comparison\n")
        f.write("- **C Reference**: Simple sparse dot product argmax operation\n")
        f.write("- **PyTorch Core**: Equivalent argmax with GPU acceleration and bound handling\n")
        f.write("- **PyTorch Full**: Complete method including data preparation and LRU updates\n\n")
        
        f.write("### Performance Notes\n")
        f.write("- Core argmax times provide the fairest comparison between implementations\n")
        f.write("- PyTorch includes additional bound calculations not present in C reference\n")
        f.write("- GPU performance depends on problem size and memory transfer overhead\n")
    
    print(f"Report saved to {md_file}")
    print(f"CSV data saved to {csv_file}")

if __name__ == "__main__":
    main()