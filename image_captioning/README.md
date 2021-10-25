# Image captioning experiments

## Requirements

- Python 3.7
- Pytorch 1.7.1
- albumentations 0.5.2
- tqdm
- sacrebleu 1.5.1
- transformers 4.6.1
- segmentation_models_pytorch 0.1.3
- pandas 1.2.4
- scikit-learn 0.24.2

```
conda create -n lsp python=3.7
conda activate lsp
conda install pytorch=1.7.1 torchvision cudatoolkit=11.0 -c pytorch
pip install tqdm albumentations==0.5.2 sacrebleu==1.5.1 transformers==4.6.1 segmentation_models_pytorch==0.1.3 pandas==1.2.4 scikit-learn==0.24.2
```

## Preparing

Prepare the datasets according to [data/README.md](data).

## Reproducing results

### CLEVR object description generation task

The user is encourgaed to read/alter the `train_clevr_run.py` file to suit their needs. 

```
python train_clevr_run.py
```

#### Running many jobs at the same time

The file `mlkitenv.json` dictates how many GPUs are available and how many jobs are allowed to run at a single moment.

```
{
    "cuda": [
        0
    ],
    "global_lock": 1,
    "num_workers": 8
}
```

This indicates that only `CUDA:0` is available, and there can be 1 job at a single moment. 

### MIMIC-CXR chest radiograph report generation task

The user is encourgaed to read/alter the `train_mimic_run.py` file to suit their needs. 

```
python train_mimic_run.py
```

#### Run RM+MCLN baseline

Follow instructions in [baselines/RM_MCLN](baselines/RM_MCLN). After the model is trained and predictions are made, you can run `eval_rm_mcln.py` to see the result metrics.

### How to observe the training stats?

The training stats are kept in `save` directory with subdirectories according to their names along with the checkpoints.

### Where to see the final results?

All finished runs will generate result metrics (in .csv) in `eval` directory with subdirectories according to their names.
