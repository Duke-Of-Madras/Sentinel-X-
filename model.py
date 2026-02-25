"""
model.py  ──  Sentinel-X | Layer 2: The AI Brain
═══════════════════════════════════════════════════════════════
Defines, trains, and exports the Autoencoder neural network.

Architecture:
  Input  [4]  →  Encoder [2 latent]  →  Decoder [4 output]

Training Logic:
  The model is trained exclusively on "safe" system behaviour.
  During inference, NORMAL input → low MSE loss (well reconstructed).
  MALICIOUS input → high MSE loss (pattern never seen during training).

Quantization:
  Simulates INT8 post-training quantization via torch.quantization.
  On AMD target hardware this would be replaced by Vitis AI quantizer
  (vai_q_pytorch) to produce a genuinely INT8 ONNX for NPU execution.

Export:
  Produces sentinel_brain.onnx in the same directory.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format="[MODEL] %(message)s")
log = logging.getLogger(__name__)

ONNX_PATH = os.path.join(os.path.dirname(__file__), "sentinel_brain.onnx")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "sentinel_brain.pt")


# ─── Autoencoder Definition ────────────────────────────────────────────────────
class SentinelAutoencoder(nn.Module):
    """
    Lightweight 4→2→4 Autoencoder.

    Encoder compresses 4 telemetry features into a 2-dimensional
    latent representation of "normal" system behaviour.
    Decoder attempts to reconstruct the original 4 features.
    High reconstruction error = anomaly = potential threat.
    """

    def __init__(self):
        super(SentinelAutoencoder, self).__init__()

        # Encoder: 4 → 2  (bottleneck / latent space)
        self.encoder = nn.Sequential(
            nn.Linear(4, 3),
            nn.ReLU(),
            nn.Linear(3, 2),
            nn.ReLU(),
        )

        # Decoder: 2 → 4  (reconstruction)
        self.decoder = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 4),
            nn.Sigmoid(),   # Output clamped to [0,1] — matches normalised input
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        return self.decoder(latent)


# ─── Training ──────────────────────────────────────────────────────────────────
def generate_safe_training_data(n_samples: int = 1000) -> torch.Tensor:
    """
    Synthesise training samples representative of 'safe' normalised telemetry.
    Values are drawn from a Gaussian centred on low-usage normal operation
    and clipped to [0, 1].

    Safe zone approximation (normalised):
      cpu_power    ~  0.10  (10-20W idle laptop)
      disk_write   ~  0.05  (minimal background write)
      net_out      ~  0.08  (idle browsing)
      api_call_freq ~ 0.15  (normal process switching)
    """
    means = np.array([0.12, 0.05, 0.08, 0.15], dtype=np.float32)
    stds  = np.array([0.04, 0.03, 0.03, 0.05], dtype=np.float32)

    data = np.random.normal(loc=means, scale=stds, size=(n_samples, 4)).astype(np.float32)
    data = np.clip(data, 0.0, 1.0)
    return torch.tensor(data)


def train_model(epochs: int = 600, lr: float = 3e-3) -> SentinelAutoencoder:
    """Train the autoencoder on safe telemetry and return the fitted model."""
    log.info("Generating safe-behaviour training dataset (2000 samples)...")
    train_data = generate_safe_training_data(n_samples=2000)

    model     = SentinelAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.3)

    log.info(f"Training Autoencoder for {epochs} epochs...")
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = model(train_data)
        loss   = criterion(output, train_data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0:
            log.info(f"  Epoch [{epoch:>4}/{epochs}]  Loss: {loss.item():.6f}  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Validation — confirm safe reconstruction error is below threshold
    model.eval()
    with torch.no_grad():
        val_data   = generate_safe_training_data(n_samples=500)
        val_output = model(val_data)
        val_loss   = criterion(val_output, val_data).item()
    log.info(f"Validation loss on safe data: {val_loss:.6f}  (threat threshold: 0.10)")
    if val_loss < 0.08:
        log.info("✅ Model converged well — safe data is well below detection threshold.")
    else:
        log.warning(f"Safe loss ({val_loss:.4f}) may be too close to threshold. Consider more epochs.")

    model.train()  # Reset back to train mode
    log.info("Training complete.")
    return model


# ─── INT8 Quantization (Simulation) ───────────────────────────────────────────
def apply_quantization_sim(model: SentinelAutoencoder) -> nn.Module:
    """
    Simulates post-training INT8 quantization via torch.quantization.

    NOTE (AMD NPU Deployment):
      In production, this step is replaced by Vitis AI quantizer
      (vai_q_pytorch) which generates a genuine INT8 ONNX model
      optimised for the XDNA 2 DPU on the Ryzen AI 400 NPU.
      The NPU's INT8 inference path delivers ~10x throughput vs FP32 CPU.
    """
    log.info("Applying INT8 quantization simulation (torch.quantization)...")

    try:
        # Fuse eligible Conv-BN-ReLU patterns (none here, but best practice)
        model_fp32_prepared = torch.quantization.prepare(
            model,
            inplace=False
        )

        # Calibration pass with representative safe data
        calibration_data = generate_safe_training_data(n_samples=200)
        with torch.no_grad():
            model_fp32_prepared(calibration_data)

        # Convert to quantized model (INT8 weights, activations)
        model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
        log.info("INT8 quantization simulation complete.")
        return model_int8

    except Exception as qe:
        # Quantization is a simulation step — if the torch version doesn't
        # support static quant on Linear layers cleanly, log and continue.
        log.warning(f"Quantization sim skipped ({qe}). Continuing with FP32 model.")
        log.info("[NOTE] On AMD target: vai_q_pytorch would generate genuine INT8 ONNX.")
        return model

    # ⚠️  We export the ORIGINAL FP32 model to ONNX for compatibility with
    #     onnxruntime CPUExecutionProvider in the PoC.
    #     On real AMD hardware, vai_q_pytorch exports to INT8 ONNX directly.


# ─── ONNX Export ───────────────────────────────────────────────────────────────
def export_to_onnx(model: SentinelAutoencoder, path: str = ONNX_PATH):
    """Export the trained FP32 model to ONNX format for NPU inference."""
    log.info(f"Exporting model to ONNX: {path}")

    model.eval()
    dummy_input = torch.zeros(1, 4)   # Batch=1, Features=4

    # Use the legacy TorchScript-based exporter (stable, no onnxscript dependency).
    # On AMD target, replace with vai_q_pytorch for genuine INT8 ONNX.
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["telemetry"],
            output_names=["reconstruction"],
            dynamic_axes={
                "telemetry":      {0: "batch_size"},
                "reconstruction": {0: "batch_size"},
            },
        )
    log.info(f"sentinel_brain.onnx saved successfully → {path}")


# ─── Entry point: run directly to train + export ──────────────────────────────
if __name__ == "__main__":
    import warnings

    # Step 1: Train autoencoder on safe telemetry
    model = train_model(epochs=200)

    # Step 2: Save raw PyTorch checkpoint (optional)
    torch.save(model.state_dict(), MODEL_PATH)
    log.info(f"PyTorch checkpoint saved → {MODEL_PATH}")

    # Step 3: Quantization simulation (INT8 calibration demonstration)
    model.eval()
    try:
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    except Exception:
        pass   # qconfig not required for PoC export
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        apply_quantization_sim(model)

    # Step 4: Export trained FP32 model to ONNX (for onnxruntime PoC)
    export_to_onnx(model, ONNX_PATH)
    print("\n✅  Model trained and exported. Ready for Sentinel-X deployment.\n")
