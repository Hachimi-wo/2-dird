# DIRD report packages

- `refined_target_negative/`: results from the mentor-refined pipeline. For each endpoint, every target-label 0 sample is treated as negative.
- `comparison_all_four_negative/`: comparison pipeline using the same deduplication and model settings, but retaining only target positives or all-four-labels-zero negatives.
- `primary_raw_run/`: canonical data products from the raw-data run. Raw source files remain under `data/raw_data`; intermediate files and the cleanroom backup are retained for provenance.

The comparison metrics are in `comparison_all_four_negative/Label_ILD_old_vs_endpoint_clean.csv`.
