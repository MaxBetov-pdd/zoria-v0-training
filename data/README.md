# Zoria MVP data preparation

This directory contains the initial source manifest. On the VPS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U datasets huggingface_hub pyyaml tqdm
python scripts/prepare_mvp_data.py
```

The script uses streaming for public datasets and writes
`data/processed/zoria_v0_train.jsonl`,
`data/processed/zoria_v0_audit_holdout.jsonl` and a manifest. It does not
download the complete Kobza corpus.

Before training:

1. inspect every source and its license;
2. remove benchmark contamination;
3. keep train/validation/private holdout separate;
4. record dataset revisions and hashes;
5. run the baseline before using the processed data.
