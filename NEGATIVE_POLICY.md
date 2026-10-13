# Negative-label policy: all-four-zero

For each endpoint, the training set keeps:

```text
target label == 1
OR
Label_ILD == Label_Edema == Label_Pleural == Label_Embolism == 0
```

A drug positive for another endpoint is excluded from the target endpoint's negative class. This policy is enabled by default in `ml_experiments/01_data_splitting.py`.

To reproduce the legacy refined target-zero policy instead, pass:

```bash
python ml_experiments/01_data_splitting.py --input both --label-col Label_ILD --target-zero-negatives
```
