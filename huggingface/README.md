# Hugging Face release

After the Colab pilot has completed baseline and adapter evaluation:

```bash
huggingface-cli login
python scripts/upload_hf_adapter.py \
  --adapter-dir outputs/zoria-qwen35-9b \
  --repo-id MaxBetov-pdd/zoria-qwen35-9b
```

Before making the model public:

- fill the model card with measured results;
- attach the final data manifest;
- verify dataset licenses;
- include the exact base-model revision;
- publish raw evaluation JSON;
- avoid leaderboard claims that cannot be reproduced.
