# -*- coding: utf-8 -*-
import sys

import torch

sys.path.append("../../")

from swift.llm import sft_main


if __name__ == '__main__':
    output = sft_main()



"""
--model_type internvl2-1b --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B --dataset coco-en-2-mini --max_length 4096 --sft_type lora
--model_type internvl2-1b --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B --dataset coco-en-2-mini --max_length 4096 --sft_type full
"""

"""
# 支持多个数据集
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train.jsonl,/mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--max_length 4096 --sft_type lora
"""

"""
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val.jsonl
--max_length 2500 --sft_type full
--num_train_epochs 20
--save_steps 200
--save_strategy steps
--save_total_limit 5
--patience 5
"""


# 自定义template

"""
--model_type internvl2-1b 
--model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B 
--dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/train_with_angle.jsonl
--val_dataset /mnt/g/dongyongfei786/custom-swift/examples/data_processing/output/val_with_angle.jsonl
--max_length 2500 --sft_type full
--num_train_epochs 20
--save_steps 200
--save_strategy steps
--save_total_limit 5
--template_type internvl2-angle
"""

'''
  --model_type internvl2-1b
  --model_id_or_path /mnt/n/model/OpenGVLab/InternVL2-1B
  --output_dir /mnt/n/model/sft-model/internvl2-1b-trainticket
  --dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_train.jsonl
  --val_dataset /mnt/n/data/mllm-data/mllm-finetune-data/trainticket/swift_label/metadata_val.jsonl
  --max_length 1024
  --gradient_accumulation_steps 16
  --eval_steps 20
  --save_steps 200
  --num_train_epochs 10
  --batch_size 1
  --learning_rate 3e-5
  --lr_scheduler_type cosine
  --sft_type lora
  --lora_target_modules ALL
  --predict_with_generate True
  --warmup_ratio 0.05
'''

# grounding
"""
  --model_type qwen2-vl-2b-instruct
  --model_id_or_path /mnt/n/model/Qwen/Qwen2-VL-2B-Instruct
  --output_dir /mnt/n/model/sft-model/refcoco-sft
  --sft_type lora
  --lora_target_modules ALL
  --dataset refcoco-unofficial-grounding#20000
"""