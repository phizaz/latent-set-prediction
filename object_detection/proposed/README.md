# Set Prediction in the Latent Space (Object detection)

## Setup

To run the experiment, you have to set up the environment by building the docker: 
```
docker build -t docker/ detr
```
Then, mount into the docker and go to the working directory using the following command:
```
docker run --gpus all -it -v deep-latent-set-prediction/object_detection/:/work/ detr
cd /work/proposed
```

## Reproducing results

To reproduce the result of LSP (GCR, d = 0.0001) with batch size = 8 run:
```bash
  python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path dataset_path --lr_drop 120 --epochs 160  --mode gcr --safe_coef 0.0001 --batch_size 8 --output_dir output_path/
```
The `dataset_path` is the path to the dataset (`../data`).

To reproduce the result of LSP (GC) run:
```bash
  python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path dataset_path --lr_drop 120 --epochs 160  --mode gc --batch_size 8 --output_dir output_path
```

All finished runs will generate result in `output_path` directory.

To run the experiment for `n` iterations with the batch_size of `b`, the options `--epochs` and `--lr_drop` should be set to  `n / (5000 / b)`, and `0.75 * n / (5000 / b)` respectively.
