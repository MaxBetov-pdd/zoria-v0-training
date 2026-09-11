# Training configs

`qwen3.8-flash-next-qlora-4xa100.yaml` is the primary Zoria v0 text-only
QLoRA configuration for the selected large VPS. It follows the official
Axolotl targets for Flash-Next:

- `ple_cpu_offload: true`
- `quantize_moe_experts: true`
- `attn_implementation: sdpa`
- explicit Gated DeltaNet targets
- no blanket `lora_target_linear`

`qwen3.8-flash-next-qlora-4xa100-smoke.yaml` is the short FSDP2 smoke test
for the selected large VPS. It is intentionally limited to 20 steps.

The current Axolotl Flash-Next integration was tested on 1xB300, and the
upstream change notes that multi-GPU was not tested. Do not extend the
4xA100 run until the smoke test completes and the loss/checkpoint are valid.

After the 20-step test succeeds:

1. use the primary config with `max_steps: 300`;
2. make rank 32 and rank 128 copies;
3. compare Ukrainian leaderboard subset and retention tests;
4. train only the best configuration for the final MVP.
