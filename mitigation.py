"""
mitigation.py  ──  Sentinel-X | Layer 4: Threat Mitigation (The Enforcer)
══════════════════════════════════════════════════════════════════════════════
Implements the kill-switch triggered when reconstruction error > threshold.

Real AMD Deployment:
  On Ryzen AI 400, this module would issue an interrupt to the AMD Secure
  Processor (ASP / PSP) via the kernel driver (amdgpu / amdfwflash).
  The ASP can hard-terminate processes with ring-0 privileges in < 1ms,
  bypassing any user-space hooks the malware may have installed.

PoC Behaviour:
  1. Spawn wannacry_simulation.py (once, on first threat detection)
  2. Suspend the process via psutil.Process.suspend()
  3. Print the neutralisation banner
"""

import os
import sys
import subprocess
import time
import psutil
import logging

logging.basicConfig(level=logging.INFO, format="[ENFORCER] %(message)s")
log = logging.getLogger(__name__)

# Threshold — tuned against training set; loss > 0.1 = anomalous
THREAT_THRESHOLD: float = 0.10

# Path to the dummy malware process
MALWARE_SCRIPT = os.path.join(os.path.dirname(__file__), "wannacry_simulation.py")

# ANSI colour helpers
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


class ThreatEnforcer:
    """
    Manages the lifecycle of the simulated malware process and kill-switch.
    Spawns wannacry_simulation.py on first threat and suspends it.
    """

    def __init__(self):
        self._malware_proc: subprocess.Popen | None = None
        self._malware_psutil: psutil.Process | None = None
        self._is_suspended = False

    def _spawn_malware_process(self):
        """Launch the dummy malware in the background (once)."""
        if self._malware_proc is not None:
            return  # Already spawned

        log.info("Spawning simulated malware process (wannacry_simulation.py)...")
        self._malware_proc = subprocess.Popen(
            [sys.executable, MALWARE_SCRIPT],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        time.sleep(0.3)   # Give it time to appear in process list
        try:
            self._malware_psutil = psutil.Process(self._malware_proc.pid)
            log.info(
                f"Malware process spawned | PID: {self._malware_proc.pid} | "
                f"Name: {self._malware_psutil.name()}"
            )
        except psutil.NoSuchProcess:
            log.error("Malware process failed to start.")

    def mitigate_threat(self, loss: float):
        """
        Entry point called by the main loop when loss > THREAT_THRESHOLD.

        Steps:
          1. Spawn malware process if not yet running
          2. Suspend it via psutil (simulates ASP hard-kill)
          3. Print neutralisation banner
        """
        # Spawn the dummy process only once per session
        self._spawn_malware_process()

        if self._malware_psutil is None:
            log.error("Cannot suspend: malware process not available.")
            return

        pid  = self._malware_proc.pid
        name = "wannacry_simulation.py"

        # ── Suspend the process ─────────────────────────────────────────────
        if not self._is_suspended:
            try:
                self._malware_psutil.suspend()
                self._is_suspended = True
                log.info(f"Process PID {pid} ({name}) suspended successfully.")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                log.warning(f"Suspend failed: {e}")

        # ── Print neutralisation banner ─────────────────────────────────────
        self._print_neutralisation_banner(pid, name, loss)

    def resume_malware(self):
        """Resume the suspended process (used when going back to SAFE mode)."""
        if self._malware_psutil and self._is_suspended:
            try:
                self._malware_psutil.resume()
                self._is_suspended = False
                log.info("Process resumed (SAFE mode restored).")
            except Exception as e:
                log.warning(f"Resume failed: {e}")

    def cleanup(self):
        """Terminate the dummy malware process on exit."""
        if self._malware_proc:
            try:
                self._malware_proc.terminate()
                self._malware_proc.wait(timeout=3)
                log.info("Malware simulation process terminated (cleanup).")
            except Exception:
                pass

    @staticmethod
    def _print_neutralisation_banner(pid: int, name: str, loss: float):
        print(f"\n{RED}{BOLD}{'═'*60}{RESET}")
        print(f"{RED}{BOLD}  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗      ██╗  ██╗{RESET}")
        print(f"{RED}{BOLD}  [!!! AMD SENTINEL-X — THREAT NEUTRALIZED !!!]{RESET}")
        print(f"{RED}{BOLD}{'═'*60}{RESET}")
        print(f"{YELLOW}  ► Reconstruction Error : {loss:.6f}  (Threshold: {THREAT_THRESHOLD}){RESET}")
        print(f"{YELLOW}  ► Target PID           : {pid}{RESET}")
        print(f"{YELLOW}  ► Target Process       : {name}{RESET}")
        print(f"{YELLOW}  ► Action               : AMD SECURE PROCESSOR (ASP) INTERRUPT ISSUED{RESET}")
        print(f"{RED}{BOLD}  [SENTINEL-X] THREAT NEUTRALIZED VIA NPU INTERRUPT.{RESET}")
        print(f"{RED}{BOLD}{'═'*60}{RESET}\n")
