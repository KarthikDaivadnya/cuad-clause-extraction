# data/

This folder holds the CUAD dataset. It's git-ignored because the full
dataset is ~380MB — don't want to commit it.

## Quick start

```bash
python scripts/download_cuad.py --sample 50
```

This downloads `CUAD_v1.zip` from Zenodo, extracts it here, and copies 50
contracts into `data/full_contract_pdf_sample/` — ready for:

```bash
python main.py --data-dir data/full_contract_pdf_sample --limit 50
```

## Manual download (if the script's network access is blocked)

1. Download `CUAD_v1.zip` from https://zenodo.org/records/4595826
   (mirror: https://huggingface.co/datasets/theatticusproject/cuad-qa)
2. Unzip it into this folder, so you have `data/CUAD_v1/full_contract_pdf/...`
3. Run `python scripts/download_cuad.py --sample 50 --skip-download` to build
   the 50-contract sample folder from your local extract.

## Expected structure after setup

```
data/
├── CUAD_v1/
│   ├── full_contract_pdf/       # ~510 contracts, organized by category
│   ├── full_contract_txt/       # plain-text versions (unused by this project)
│   ├── CUAD_v1.json             # original expert clause annotations
│   └── master_clauses.csv
└── full_contract_pdf_sample/    # the 50-contract subset main.py reads by default
```
