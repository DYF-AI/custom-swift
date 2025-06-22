# Experimental environment: A100
# 80GB GPU memory
#CUDA_VISIBLE_DEVICES=0 swift sft \
#    --model_type internvl2-1b \
#    --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B \
#    --dataset coco-en-2-mini \
#    --max_length 4096


# ms-swift-3.0
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model /mnt/n/model/OpenGVLab/InternVL2-1B  \
    --dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train_with_angle.jsonl \
    --val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val_with_angle.jsonl \
    --max_length 2500 \
    --train_type full \
    --num_train_epochs 1 \
    --save_steps 200 \
    --save_strategy steps \
    --save_total_limit 5 \
    --template internvl2-angle