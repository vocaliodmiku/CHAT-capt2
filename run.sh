/ssd3/other/penglinkai01/miniconda3/envs/pytorch/bin/deepspeed \
    --master_port 10900 \
    --num_gpus 8 \
    main.py \
    --model_name_or_path /ssd9/exec/penglinkai/seamless_communication/ckpts/7B  \
    --per_device_train_batch_size 1 \
    --train_datasets /ssd9/exec/penglinkai/seamless_communication/data/asr \
    --eval_datasets /ssd9/exec/penglinkai/seamless_communication/data/asr \
    --tf32 True \
    --bf16 True \
    --trust_remote_code True \
    --output_dir /ssd9/exec/penglinkai/chatcapt/output \
    --need_eval \
    --eval_strategy steps \
    --eval_interval 10000 \
    --max_length 1024 


#     --zero_stage 3  可用