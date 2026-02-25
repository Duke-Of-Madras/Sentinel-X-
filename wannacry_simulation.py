"""
wannacry_simulation.py
──────────────────────────────────────────────────────────
Dummy process used exclusively by Sentinel-X for demonstration.
This script mimics a benign "sleeping" process that the kill-switch
suspends when the NPU detects anomalous reconstruction error.

DO NOT actually run this directly during demos — Sentinel-X spawns it.
"""

import time
import sys

if __name__ == "__main__":
    print("[wannacry_simulation] Process started. Simulating malicious activity...")
    while True:
        time.sleep(0.1)   # Idle loop — safe, no actual harm
