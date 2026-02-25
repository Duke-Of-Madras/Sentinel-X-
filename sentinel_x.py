"""
sentinel_x.py  ──  Sentinel-X | Main Orchestrator
═══════════════════════════════════════════════════════════════════════════════
Entry point for the Sentinel-X autonomous malware detection system.

Execution Flow:
  1. INIT     – Load sentinel_brain.onnx onto VitisAI EP (or CPU fallback)
  2. LOOP     – Every 0.5 s, collect telemetry → run NPU inference → compute loss
  3. DISPLAY  – Print colour-coded live dashboard (green=safe, red=threat)
  4. DECIDE   – If loss > 0.10 → trigger kill-switch via ThreatEnforcer
  5. TRIGGER  – Press  M  (or Enter) to manually inject malicious telemetry

Controls (keyboard, non-blocking):
  M / m / Enter  →  Toggle ATTACK mode (inject malicious telemetry spike)
  Q / q / Ctrl+C →  Quit cleanly

Dashboard Colours:
  GREEN   [NPU: ACTIVE] System Normal  | live metrics
  RED     [!!! THREAT DETECTED !!!]    | reconstruction error spike
  YELLOW  [ACTION] ASP Interrupt Sent  | process kill confirmation
"""

import os
import sys
import time
import threading
import logging
import numpy as np

# ── Sentinel-X sub-modules ────────────────────────────────────────────────────
from data_ingestion import get_safe_telemetry, get_malicious_telemetry
from npu_engine      import NPUEngine
from mitigation      import ThreatEnforcer, THREAT_THRESHOLD

# ── Logging (suppress sub-module noise in dashboard mode) ─────────────────────
logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(message)s")

ONNX_PATH = os.path.join(os.path.dirname(__file__), "sentinel_brain.onnx")

# ─── ANSI helpers ─────────────────────────────────────────────────────────────
GRN   = "\033[92m"
RED   = "\033[91m"
YLW   = "\033[93m"
CYN   = "\033[96m"
GRY   = "\033[90m"
BOLD  = "\033[1m"
DIM   = "\033[2m"
RST   = "\033[0m"

LOOP_INTERVAL = 0.5        # seconds between telemetry captures
ATTACK_CYCLES = 6          # how many cycles to stay in ATTACK mode after trigger


# ─── Non-blocking keyboard listener ───────────────────────────────────────────
class KeyListener:
    """
    Runs in a daemon thread.
    Sets `attack_triggered` flag when the user presses M / Enter.
    Sets `quit_flag` when Q / Ctrl+C is pressed.
    """

    def __init__(self):
        self.attack_triggered = threading.Event()
        self.quit_flag        = threading.Event()
        self._thread          = threading.Thread(target=self._listen, daemon=True)

    def start(self):
        self._thread.start()

    def _listen(self):
        try:
            # Windows: use msvcrt for non-blocking key reads
            if sys.platform == "win32":
                import msvcrt
                while not self.quit_flag.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch().lower()
                        if ch in ("m", "\r", "\n"):
                            self.attack_triggered.set()
                            print(f"\n{YLW}{BOLD}  [KEY] Attack mode triggered by user!{RST}")
                        elif ch in ("q",):
                            self.quit_flag.set()
                    time.sleep(0.05)
            else:
                # Unix / macOS: use tty + termios
                import tty, termios, select
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    while not self.quit_flag.is_set():
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                        if rlist:
                            ch = sys.stdin.read(1).lower()
                            if ch in ("m", "\r", "\n"):
                                self.attack_triggered.set()
                                print(f"\n{YLW}{BOLD}  [KEY] Attack mode triggered by user!{RST}")
                            elif ch in ("q", "\x03"):
                                self.quit_flag.set()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except Exception:
            # If keyboard listener fails, still run without it
            pass


# ─── Terminal dashboard ────────────────────────────────────────────────────────
def print_banner():
    """One-time startup banner."""
    os.system("cls" if sys.platform == "win32" else "clear")
    print(f"{CYN}{BOLD}")
    print("  ╔══════════════════════════════════════════════════════════════╗")
    print("  ║          S E N T I N E L - X   //  AMD RYZEN AI 400         ║")
    print("  ║        Autonomous NPU-Accelerated Malware Detection PoC      ║")
    print("  ║                 XDNA™ 2  |  Vitis AI EP  |  ONNX RT          ║")
    print("  ╚══════════════════════════════════════════════════════════════╝")
    print(f"{RST}")
    print(f"  {GRY}Controls:  [ M ] or [ Enter ] → Inject Malicious Telemetry{RST}")
    print(f"  {GRY}           [ Q ]               → Quit{RST}")
    print()


def fmt_metric(label: str, value: float, unit: str, width: int = 22) -> str:
    return f"{label:<{width}} {CYN}{value:>8.2f} {unit}{RST}"


def print_safe_frame(cycle: int, loss: float, raw: dict, provider: str):
    ep_tag = (
        f"{GRN}[NPU: ACTIVE]{RST}"
        if provider == "VitisAIExecutionProvider"
        else f"{YLW}[CPU SIM MODE]{RST}"
    )
    print(
        f"\r  {ep_tag}  "
        f"Cycle {cycle:>5}  |  "
        f"{GRN}System: SAFE{RST}  |  "
        f"Power: {raw.get('cpu_power_w', 0):.1f}W  |  "
        f"Disk: {raw.get('disk_write_mbps', 0):.1f} MB/s  |  "
        f"Net: {raw.get('net_out_mbps', 0):.1f} MB/s  |  "
        f"API: {raw.get('api_call_freq', 0):.0f}/s  |  "
        f"Loss: {GRN}{loss:.4f}{RST}",
        end="",
        flush=True,
    )


def print_threat_frame(cycle: int, loss: float, raw: dict, provider: str):
    ep_tag = (
        f"{GRN}[NPU: ACTIVE]{RST}"
        if provider == "VitisAIExecutionProvider"
        else f"{YLW}[CPU SIM MODE]{RST}"
    )
    print(
        f"\n  {ep_tag}  "
        f"Cycle {cycle:>5}  |  "
        f"{RED}{BOLD}[!!! THREAT DETECTED !!!]{RST}  |  "
        f"Power: {RED}{raw.get('cpu_power_w', 0):.1f}W{RST}  |  "
        f"Disk: {RED}{raw.get('disk_write_mbps', 0):.1f} MB/s{RST}  |  "
        f"Net: {RED}{raw.get('net_out_mbps', 0):.1f} MB/s{RST}  |  "
        f"API: {RED}{raw.get('api_call_freq', 0):.0f}/s{RST}  |  "
        f"Reconstruction Error: {RED}{BOLD}{loss:.4f}{RST}"
    )
    print(
        f"  {YLW}[ACTION]{RST} ASP Interrupt Sent. "
        f"Terminating PID via Secure Processor → "
        f"{YLW}wannacry_simulation.py{RST}"
    )


# ─── Main Loop ────────────────────────────────────────────────────────────────
def main():
    print_banner()

    # Step 1: Load model onto NPU (or CPU fallback)
    print(f"  {CYN}Initialising NPU Engine...{RST}")
    try:
        engine = NPUEngine(ONNX_PATH)
    except FileNotFoundError as e:
        print(f"\n  {RED}ERROR: {e}{RST}")
        print(f"  Run:  python model.py   (in the sentinel_x/ directory)\n")
        sys.exit(1)

    provider = engine.provider
    enforcer = ThreatEnforcer()
    listener = KeyListener()
    listener.start()

    ep_label = (
        f"{GRN}VitisAI Execution Provider (AMD XDNA 2 NPU){RST}"
        if engine.is_npu_active
        else f"{YLW}CPUExecutionProvider (simulation / no NPU driver){RST}"
    )
    print(f"  Execution Provider: {ep_label}")
    print(f"  Threat Threshold:   {THREAT_THRESHOLD}")
    print(f"  Poll Interval:      {LOOP_INTERVAL}s")
    print(f"\n  {GRN}[SENTINEL-X ARMED — monitoring started]{RST}\n")
    time.sleep(1.0)

    cycle         = 0
    attack_cycles_left = 0

    try:
        while not listener.quit_flag.is_set():
            cycle += 1

            # ── Determine telemetry mode ─────────────────────────────────
            if listener.attack_triggered.is_set():
                listener.attack_triggered.clear()
                attack_cycles_left = ATTACK_CYCLES

            in_attack_mode = (attack_cycles_left > 0)
            if in_attack_mode:
                attack_cycles_left -= 1

            # ── Layer 1: Collect telemetry ───────────────────────────────
            if in_attack_mode:
                telemetry, raw = get_malicious_telemetry()
            else:
                telemetry, raw = get_safe_telemetry()

            # ── Layer 3: NPU Inference ───────────────────────────────────
            reconstruction = engine.infer(telemetry)

            # ── Layer 3b: Compute reconstruction loss ────────────────────
            loss = engine.compute_loss(telemetry, reconstruction)

            # ── Layer 4: Decision & Dashboard ───────────────────────────
            if loss > THREAT_THRESHOLD:
                print_threat_frame(cycle, loss, raw, provider)
                enforcer.mitigate_threat(loss)
            else:
                # Gradually resume if we come back to safe mode
                if enforcer._is_suspended:
                    enforcer.resume_malware()
                print_safe_frame(cycle, loss, raw, provider)

            time.sleep(LOOP_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n\n  {GRY}Shutting down Sentinel-X...{RST}")
        enforcer.cleanup()
        print(f"  {GRN}Sentinel-X terminated cleanly. Goodbye.{RST}\n")


if __name__ == "__main__":
    main()
