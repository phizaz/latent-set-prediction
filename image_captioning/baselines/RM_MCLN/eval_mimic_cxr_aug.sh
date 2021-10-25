python main.py \
--image_dir data/images512/ \
--ann_path data/annotation_ours2.json \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 64 \
--epochs 30 \
--save_dir results/mimic_cxr2_aug \
--resume results/mimic_cxr2_aug/model_best.pth \
--eval 1 \
--step_size 1 \
--gamma 0.8 \
--monitor_metric loss \
--monitor_mode min \
--seed 456789 \
--aug=ours
