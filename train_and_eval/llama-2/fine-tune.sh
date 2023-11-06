CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=10044 train.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --data_path squad/Preprocessed_train-v1.1.json \
    --bf16 True \
    --output_dir squad-llama-2-7B \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'