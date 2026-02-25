# Sentinel-X 🛡️

**Autonomous, NPU-Accelerated Malware Detection — AMD Ryzen AI 400 (XDNA™ 2) PoC**

---

## What is Sentinel-X?

Sentinel-X is a real-time security system that uses an **Autoencoder neural network** to detect anomalous hardware behaviour. It was trained exclusively on "safe" system telemetry. When malware (ransomware, cryptominers, spyware) runs, it drives metrics outside the safe distribution — the autoencoder fails to reconstruct them, producing a **high MSE loss**, which triggers the kill-switch.

```
Telemetry [1,4]  →  AMD NPU (XDNA 2)  →  Reconstruction  →  MSE Loss  →  Kill Switch
```

---

## Project Structure

```
sentinel_x/
├── sentinel_x.py           # Main orchestrator (run this)
├── data_ingestion.py       # Layer 1 – Hardware telemetry simulation
├── model.py                # Layer 2 – Autoencoder + ONNX export
├── npu_engine.py           # Layer 3 – ONNX Runtime / Vitis AI EP
├── mitigation.py           # Layer 4 – Threat kill-switch
├── wannacry_simulation.py  # Dummy malware process (spawned by kill-switch)
├── requirements.txt        # Python dependencies
└── sentinel_brain.onnx     # Generated after running model.py
```

---

## Hardware Target

| Component | Spec |
|-----------|------|
| SoC | AMD Ryzen AI 400 (Strix Point) |
| NPU | XDNA™ 2 Architecture — 50 TOPS |
| Software | Ryzen AI Software 1.7 + ROCm 7.2 |
| Runtime | ONNX Runtime with VitisAI Execution Provider |
| Precision | INT8 (simulated via `torch.quantization`) |

> **Non-AMD machines:** Automatically falls back to `CPUExecutionProvider`. All functionality works.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model & Export ONNX

```bash
python model.py
```

Expected output:
```
[MODEL] Generating safe-behaviour training dataset...
[MODEL] Training Autoencoder for 200 epochs...
[MODEL] Epoch [  50/200]  Loss: 0.002341
...
[MODEL] sentinel_brain.onnx saved successfully
✅  Model trained and exported. Ready for Sentinel-X deployment.
```

### 3. Launch Sentinel-X

```bash
python sentinel_x.py
```

---

## Dashboard Controls

| Key | Action |
|-----|--------|
| **M** or **Enter** | Inject malicious telemetry spike (demo trigger) |
| **Q** | Quit cleanly |
| **Ctrl+C** | Force quit |

---

## Live Dashboard

**Safe state (Green):**
```
[NPU: ACTIVE]  Cycle    42  |  System: SAFE  |  Power: 14.2W  |  Disk: 3.1 MB/s  |  Net: 0.8 MB/s  |  API: 124/s  |  Loss: 0.0023
```

**Threat detected (Red):**
```
[NPU: ACTIVE]  Cycle    43  |  [!!! THREAT DETECTED !!!]  |  Power: 89.4W  |  Disk: 431.2 MB/s  |  Net: 84.1 MB/s  |  API: 1867/s  |  Reconstruction Error: 0.8341
[ACTION] ASP Interrupt Sent. Terminating PID via Secure Processor → wannacry_simulation.py
```

---

## How the AI Works

The autoencoder learns a **compressed representation** of normal system behaviour:

```
Normal Input [4] → Encoder → Latent [2] → Decoder → Reconstruction [4]
                                                              ↓
                                                     MSE vs Original
                                                     Low = Safe ✅
                                                     High = Threat 🚨
```

Threshold: **0.10** (tuned against training distribution)

---

## AMD NPU Integration

On target hardware with **Ryzen AI Software 1.7**:

1. Replace `onnxruntime` with `onnxruntime-vitisai`
2. Provide `vaip_config.json` pointing to the XDNA 2 xclbin
3. Run `vai_q_pytorch` quantization for genuine INT8 ONNX
4. The `NPUEngine` class auto-selects `VitisAIExecutionProvider`

---



