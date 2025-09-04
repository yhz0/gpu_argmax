"""Memory monitoring utility for tracking RAM and VRAM usage."""

import psutil
import torch
import gc
from typing import Dict, Optional
import os


class MemoryMonitor:
    """Monitor RAM and VRAM usage to detect memory leaks."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.process = psutil.Process(os.getpid())
        self.baseline_ram = None
        self.baseline_vram = None
        self.measurements = []
        
    def get_ram_usage(self) -> float:
        """Get current RAM usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_vram_usage(self) -> Dict[str, float]:
        """Get current VRAM usage in MB."""
        if not torch.cuda.is_available():
            return {"allocated": 0.0, "reserved": 0.0, "free": 0.0}
        
        allocated = torch.cuda.memory_allocated(self.device) / 1024 / 1024
        reserved = torch.cuda.memory_reserved(self.device) / 1024 / 1024
        
        # Get total GPU memory
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024
        free = total_memory - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved, 
            "free": free,
            "total": total_memory
        }
    
    def set_baseline(self, label: str = "baseline"):
        """Set baseline memory usage."""
        # Force garbage collection before baseline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.baseline_ram = self.get_ram_usage()
        self.baseline_vram = self.get_vram_usage()
        
        print(f"[{label}] Baseline - RAM: {self.baseline_ram:.1f}MB, "
              f"VRAM: {self.baseline_vram['allocated']:.1f}MB allocated, "
              f"{self.baseline_vram['reserved']:.1f}MB reserved")
    
    def measure(self, label: str, force_gc: bool = False) -> Dict[str, float]:
        """Take a memory measurement and return deltas from baseline."""
        if force_gc:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        current_ram = self.get_ram_usage()
        current_vram = self.get_vram_usage()
        
        # Calculate deltas from baseline
        ram_delta = current_ram - (self.baseline_ram or 0)
        vram_allocated_delta = current_vram["allocated"] - (self.baseline_vram["allocated"] if self.baseline_vram else 0)
        vram_reserved_delta = current_vram["reserved"] - (self.baseline_vram["reserved"] if self.baseline_vram else 0)
        
        measurement = {
            "label": label,
            "ram_current": current_ram,
            "ram_delta": ram_delta,
            "vram_allocated_current": current_vram["allocated"],
            "vram_allocated_delta": vram_allocated_delta,
            "vram_reserved_current": current_vram["reserved"],
            "vram_reserved_delta": vram_reserved_delta,
            "vram_free": current_vram["free"]
        }
        
        self.measurements.append(measurement)
        
        # Print measurement
        print(f"[{label}] RAM: {current_ram:.1f}MB ({ram_delta:+.1f}), "
              f"VRAM allocated: {current_vram['allocated']:.1f}MB ({vram_allocated_delta:+.1f}), "
              f"VRAM reserved: {current_vram['reserved']:.1f}MB ({vram_reserved_delta:+.1f})")
        
        return measurement
    
    def detect_leak(self, threshold_mb: float = 100.0) -> bool:
        """Detect if there's a potential memory leak based on recent measurements."""
        if len(self.measurements) < 2:
            return False
        
        # Check last measurement vs baseline
        last = self.measurements[-1]
        ram_leak = last["ram_delta"] > threshold_mb
        vram_leak = last["vram_allocated_delta"] > threshold_mb
        
        if ram_leak or vram_leak:
            print(f"⚠️  POTENTIAL LEAK DETECTED!")
            if ram_leak:
                print(f"   RAM increased by {last['ram_delta']:.1f}MB (threshold: {threshold_mb}MB)")
            if vram_leak:
                print(f"   VRAM allocated increased by {last['vram_allocated_delta']:.1f}MB (threshold: {threshold_mb}MB)")
            return True
        
        return False
    
    def print_summary(self):
        """Print a summary of all measurements."""
        if not self.measurements:
            print("No measurements taken.")
            return
        
        print("\n=== MEMORY USAGE SUMMARY ===")
        print(f"{'Label':<20} {'RAM (MB)':<12} {'RAM Δ':<10} {'VRAM Alloc (MB)':<15} {'VRAM Alloc Δ':<12} {'VRAM Rsv (MB)':<13} {'VRAM Rsv Δ':<11}")
        print("-" * 95)
        
        for m in self.measurements:
            print(f"{m['label']:<20} {m['ram_current']:<12.1f} {m['ram_delta']:<10.1f} "
                  f"{m['vram_allocated_current']:<15.1f} {m['vram_allocated_delta']:<12.1f} "
                  f"{m['vram_reserved_current']:<13.1f} {m['vram_reserved_delta']:<11.1f}")
    
    def clear_measurements(self):
        """Clear all measurements."""
        self.measurements.clear()


# Convenience function for quick monitoring
def monitor_memory_around_function(func, *args, monitor_label="function_call", **kwargs):
    """Wrapper to monitor memory usage around a function call."""
    monitor = MemoryMonitor()
    monitor.set_baseline(f"{monitor_label}_baseline")
    
    # Before function call
    monitor.measure(f"{monitor_label}_before")
    
    # Call function
    result = func(*args, **kwargs)
    
    # After function call
    monitor.measure(f"{monitor_label}_after")
    
    # Check for leaks
    monitor.detect_leak()
    
    return result, monitor