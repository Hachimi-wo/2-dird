# Report version

This folder preserves the mentor repository layout from `DIRD_revised_7_Oct_26_(1)`.

Code is overlaid from `DIRD_revised_7_Oct_26` and includes the reproducibility fixes. The canonical cleaned dataset and phase outputs are copied from the raw-data pipeline run retained in `data/`. API cache/checkpoint files are kept as provenance; they are not substitutes for the raw inputs.

`results/` contains the existing ML outputs. The alternative endpoint-clean comparison is under `reports/comparison_all_four_negative/`.

This `_report_1` copy uses all-four-labels-zero negatives by default during split generation. See `NEGATIVE_POLICY.md`.
