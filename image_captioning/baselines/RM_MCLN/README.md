# R2Gen

This is the implementation of [Generating Radiology Reports via Memory-driven Transformer](https://arxiv.org/pdf/2010.16056.pdf) at EMNLP-2020.

The original repository is at: https://github.com/cuhksz-nlp/R2Gen

## Prepare

Follow the instructions in [data/README.md](data)

## Run on MIMIC-CXR

Run `bash run_mimic_cxr_aug.sh` to train a model on the MIMIC-CXR data.

Evaluate `bash run_mimic_cxr_aug.sh` to generate the best model.

The generation results will be kept in `results/mimic_cxr2_aug/gen_test.pkl`

You can evalute the generation results using the script outside at `eval_rm_mcln.py`.
