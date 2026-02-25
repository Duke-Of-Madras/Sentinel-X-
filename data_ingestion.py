"""
data_ingestion.py  ──  Sentinel-X | Layer 1: Data Ingestion
═══════════════════════════════════════════════════════════════
Simulates real-time Hardware Telemetry using psutil + numpy.

Metrics collected (4 features):
  [0] cpu_power_w      – CPU power draw in watts (simulated from cpu_percent)
  [1] disk_write_mbps  – Disk I/O write speed  (MB/s)
  [2] net_out_mbps     – Network outbound bandwidth (MB/s)
  [3] api_call_freq    – Normalised process API call frequency (proxy: ctx_switches)

Output:
  np.ndarray of shape [1, 4], dtype=float32 — ready for ONNX inference.

Safe Mode  → values drawn from Gaussian centred on normal operational ranges
Attack Mode → values spiked to simulate ransomware / cryptominer behaviour
"""

import numpy as np
import psutil
import time
from typing import Tuple

# ─── Normalisation bounds (min, max) per feature ──────────────────────────────
# Tuned for a mid-range laptop/desktop doing typical workload.
NORM_BOUNDS = {
    "cpu_power_w":     (5.0,  95.0),   # TDP range for modern APU (watts)
    "disk_write_mbps": (0.0,  500.0),  # MB/s (NVMe burst upper bound)
    "net_out_mbps":    (0.0,  100.0),  # MB/s (1 Gbps LAN)
    "api_call_freq":   (0.0,  2000.0), # ctx switches/s proxy
}


def _clamp_norm(value: float, lo: float, hi: float) -> float:
    """MinMax-normalise `value` to [0, 1] and clamp edges."""
    if hi == lo:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _read_live_metrics() -> dict:
    """
    Pull actual OS-level metrics via psutil.
    Returns raw (un-normalised) values.
    """
    # CPU % → rough watt estimate (assume max TDP ≈ 95W)
    cpu_pct   = psutil.cpu_percent(interval=None)
    cpu_power = 5.0 + (cpu_pct / 100.0) * 90.0        # 5W idle → 95W full load

    # Disk counters (delta since last call)
    disk_io   = psutil.disk_io_counters()
    disk_wmbs = (disk_io.write_bytes / 1e6) if disk_io else 0.0
    # Use absolute bytes then normalise; real impl would delta over interval.

    # Network counters
    net_io    = psutil.net_io_counters()
    net_ombs  = (net_io.bytes_sent / 1e6) if net_io else 0.0

    # API call frequency proxy: sum of voluntary ctx switches across top procs
    ctx_total = 0
    for proc in psutil.process_iter(["num_ctx_switches"]):
        try:
            cs = proc.info["num_ctx_switches"]
            if cs:
                ctx_total += cs.voluntary
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    api_freq = min(ctx_total, 2000.0)   # cap at normalisation max

    return {
        "cpu_power_w":     cpu_power,
        "disk_write_mbps": disk_wmbs % 500.0,  # wrap for display stability
        "net_out_mbps":    net_ombs  % 100.0,
        "api_call_freq":   api_freq,
    }


def get_safe_telemetry() -> Tuple[np.ndarray, dict]:
    """
    SAFE MODE: Blends real system metrics with a Gaussian 'normal' baseline.
    Returns a [1, 4] float32 tensor and a dict of readable values for the HUD.
    """
    live   = _read_live_metrics()
    bounds = NORM_BOUNDS

    # Blend real metrics with stable Gaussian noise to ensure values
    # stay in the "trained safe zone" the autoencoder learned.
    safe_raw = {
        "cpu_power_w":     np.clip(live["cpu_power_w"]    + np.random.normal(0, 2),  5,  40),
        "disk_write_mbps": np.clip(live["disk_write_mbps"]+ np.random.normal(0, 5),  0,  80),
        "net_out_mbps":    np.clip(live["net_out_mbps"]   + np.random.normal(0, 1),  0,  20),
        "api_call_freq":   np.clip(live["api_call_freq"]  + np.random.normal(0, 50), 0, 500),
    }

    normalised = np.array([
        _clamp_norm(safe_raw["cpu_power_w"],     *bounds["cpu_power_w"]),
        _clamp_norm(safe_raw["disk_write_mbps"], *bounds["disk_write_mbps"]),
        _clamp_norm(safe_raw["net_out_mbps"],    *bounds["net_out_mbps"]),
        _clamp_norm(safe_raw["api_call_freq"],   *bounds["api_call_freq"]),
    ], dtype=np.float32).reshape(1, 4)

    return normalised, safe_raw


def get_malicious_telemetry() -> Tuple[np.ndarray, dict]:
    """
    ATTACK MODE (manually triggered): Injects ransomware-like telemetry spikes.
    CPU pegged, disk hammered, network exfiltrating, API calls exploding.
    Returns a [1, 4] float32 tensor guaranteed to cause high reconstruction error.
    """
    attack_raw = {
        "cpu_power_w":     88.0 + np.random.normal(0, 2),    # 90W+ (cryptominer)
        "disk_write_mbps": 430.0 + np.random.normal(0, 10),  # Full NVMe encryption
        "net_out_mbps":    85.0  + np.random.normal(0, 5),   # C2 exfiltration
        "api_call_freq":   1850.0 + np.random.normal(0, 50), # Rapid VirtualAlloc/WriteFile
    }

    bounds = NORM_BOUNDS
    normalised = np.array([
        _clamp_norm(attack_raw["cpu_power_w"],     *bounds["cpu_power_w"]),
        _clamp_norm(attack_raw["disk_write_mbps"], *bounds["disk_write_mbps"]),
        _clamp_norm(attack_raw["net_out_mbps"],    *bounds["net_out_mbps"]),
        _clamp_norm(attack_raw["api_call_freq"],   *bounds["api_call_freq"]),
    ], dtype=np.float32).reshape(1, 4)

    return normalised, attack_raw
