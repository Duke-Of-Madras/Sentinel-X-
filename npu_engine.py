"""
npu_engine.py  ──  Sentinel-X | Layer 3: NPU Acceleration Engine

Loads sentinel_brain.onnx into ONNX Runtime and runs inference.

Execution Providers (priority order):
  1. VitisAIExecutionProvider  → AMD XDNA 2 NPU  (Ryzen AI 400 target)
  2. CPUExecutionProvider      → Fallback for development / non-NPU systems

On AMD Target Hardware:
  VitisAI EP routes the INT8 ONNX graph through the XDNA 2 DPU engine,
  achieving ~10 TOPS of INT8 throughput with <5ms latency per inference.
  The EP requires Ryzen AI Software 1.7 and the AMD NPU driver stack.

Reference:
  https://ryzenai.docs.amd.com/en/latest/
"""

import os
import logging
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="[NPU ENGINE] %(message)s")
log = logging.getLogger(__name__)

ONNX_PATH = os.path.join(os.path.dirname(__file__), "sentinel_brain.onnx")

# ─── Vitis AI EP configuration ────────────────────────────────────────────────
# On real AMD hardware with Ryzen AI SW 1.7, point this to the compiled
# xclbin for XDNA 2 and the quantized model config directory.
VITISAI_EP_OPTIONS = {
    "config_file":     "./vaip_config.json",  # Vitis AI EP config (hardware)
    "cacheDir":        "./cache",             # Compiled kernel cache
    "cacheKey":        "sentinel_x_cache",
}


def _build_session(onnx_path: str) -> tuple[ort.InferenceSession, str]:
    """
    Attempt to create an ONNX Runtime session preferring the AMD NPU.
    Falls back to CPU if VitisAI EP is not installed / not available.

    Returns (session, provider_name_used).
    """
    available_providers = ort.get_available_providers()
    log.info(f"Available ORT providers: {available_providers}")

    # ── Try NPU first ──────────────────────────────────────────────────────
    if "VitisAIExecutionProvider" in available_providers:
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["VitisAIExecutionProvider"],
                provider_options=[VITISAI_EP_OPTIONS],
            )
            log.info("✅  VitisAI Execution Provider ACTIVE — inference running on NPU.")
            return session, "VitisAIExecutionProvider"

        except Exception as npu_err:
            log.warning(f"VitisAI EP init failed ({npu_err}). Falling back to CPU.")

    # ── CPU Fallback ───────────────────────────────────────────────────────
    log.warning(
        "⚠️  WARNING: VitisAI EP not found or failed. "
        "Falling back to CPUExecutionProvider. "
        "For full NPU acceleration, install Ryzen AI Software 1.7 and AMD NPU drivers."
    )
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    log.info("CPUExecutionProvider ACTIVE — running in simulation mode.")
    return session, "CPUExecutionProvider"


class NPUEngine:
    """
    Wraps the ONNX Runtime session and exposes a simple `infer` method.
    Handles provider selection, session lifecycle, and output parsing.
    """

    def __init__(self, onnx_path: str = ONNX_PATH):
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"Model not found: {onnx_path}\n"
                "Run `python model.py` first to train and export the model."
            )

        self.session, self.provider = _build_session(onnx_path)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        log.info(
            f"NPUEngine ready | "
            f"Input: '{self.input_name}' [1,4] | "
            f"Provider: {self.provider}"
        )

    def infer(self, telemetry: np.ndarray) -> np.ndarray:
        """
        Run a single forward pass through the autoencoder.

        Args:
            telemetry: float32 ndarray of shape [1, 4]

        Returns:
            reconstruction: float32 ndarray of shape [1, 4]
        """
        if telemetry.dtype != np.float32:
            telemetry = telemetry.astype(np.float32)

        feed = {self.input_name: telemetry}
        outputs = self.session.run([self.output_name], feed)
        return outputs[0]   # shape [1, 4]

    def compute_loss(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Mean Squared Error between input and reconstruction."""
        return float(np.mean((original - reconstructed) ** 2))

    @property
    def is_npu_active(self) -> bool:
        return self.provider == "VitisAIExecutionProvider"
