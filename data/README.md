# Data layout

- `raw_data/`: immutable source inputs.
- `curated_data/`: curated source tables.
- `intermediate_data/`: current pipeline intermediates and API checkpoints/cache.
- `cleaned_data/`: final ML-ready dataset.
- `featurized_data/`, `split_data/`: ML preprocessing outputs.
- `*_cleanroom_backup_20261012/`: preserved dated backup; do not overwrite.
- `audit/`: hashes and provenance manifests.

Report-ready copies are under `reports/`.
