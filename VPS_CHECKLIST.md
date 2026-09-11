# VPS checklist before spending credits

## Selected offer

```text
4 × A100 80GB
48 vCPU
384GB RAM
800GB SSD
```

For Qwen3.8-Flash-Next QLoRA:

- **GPU:** enough in theory for FSDP sharding, but multi-GPU support is
  experimental for this model in Axolotl.
- **RAM:** comfortable for `ple_cpu_offload: true` and FSDP.
- **Disk:** enough for the original checkpoint, HF cache, prepared data and
  outputs if at least 600GB is free.

The quoted large price is mathematically consistent with a monthly estimate:

```text
$7.16 × 730 hours ≈ $5,226.80
```

Do not commit the full monthly amount until the 20-step smoke test succeeds.

## What to ask the provider

- Are all four A100s on the same host?
- Is NVLink available between the GPUs?
- Is the 800GB disk local NVMe or network storage?
- Is the driver/CUDA stack compatible with the current PyTorch build?
- Is the instance interruptible or preemptible?
- Is data retained when the VM is stopped?
- Are outbound traffic and attached volumes billed separately?

## Commands on the VPS

```bash
git clone <your-zoria-repository-or-copy> zoria
cd zoria
bash scripts/bootstrap_vps.sh
source .venv/bin/activate
bash scripts/preflight_4xa100.sh
bash scripts/download_and_prepare.sh
bash scripts/run_4xa100_baseline.sh
bash scripts/run_4xa100_smoke.sh
```

The preflight intentionally stops before model download when free disk is
below 600GB. Set `ZORIA_MIN_FREE_GB` only after adding a verified persistent
volume.

## Stop conditions

Stop the paid run if any of these happens:

- model files do not fit with at least 50GB free;
- only one GPU is used;
- either GPU OOMs during the 20-step test;
- loss becomes NaN;
- checkpoint cannot be saved and reloaded;
- collective operations hang;
- the model loads on a different architecture than `Qwen4Exp`.

## Fallback

If the 4-GPU path fails, use this VPS for data preparation and Qwen3.8-27B
development. Do not silently switch to an untested quantized Flash-Next
checkpoint.
