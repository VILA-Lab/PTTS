uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-7B-Instruct" # meta-llama/Llama-3.1-70B-Instruct
lr=1e-5
min_lr=0
epochs=5
micro_batch_size=1 # If 2 nodes with 8 gpus each, batch_size will be 16
push_to_hub=true
gradient_accumulation_steps=6   #this is for 7B
max_steps=-1
gpu_count=$(nvidia-smi -L | wc -l)

export WANDB_MODE=disabled
export gpu_count=$(nvidia-smi -L | wc -l)
export NUM_NODES=1
export REPLICA_RANK=0
export REPLICA_HOSTNAME=localhost
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512


torchrun \
--nnodes ${NUM_NODES}:${NUM_NODES} \
--node_rank=$REPLICA_RANK \
--nproc-per-node ${gpu_count} \
--rdzv_id=12347 \
--rdzv_backend=c10d \
--rdzv_conf='read_timeout=420' \
--rdzv_endpoint=$REPLICA_HOSTNAME:29401 \
sft.py \
--per_device_train_batch_size=${micro_batch_size} \
--per_device_eval_batch_size=${micro_batch_size} \
--gradient_accumulation_steps=${gradient_accumulation_steps} \
--train_file_path="original(90)_7B_tokonized.csv" \
--block_size=20000 \
--model_name=${base_model} \
--warmup_ratio=0.05 \
--fsdp="full_shard auto_wrap" \
--fsdp_config="fsdp_config_qwen.json" \
--bf16=True \
--eval_strategy="steps" \
--eval_steps=50 \
--logging_steps=1 \
--save_steps=100 \
--lr_scheduler_type cosine \
--learning_rate ${lr} \
--weight_decay 1e-4 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--output_dir="ckpts/m1-DeepSeekData_original90_7b_${uid}" \
--num_train_epochs ${epochs} \
--save_only_model=True \
--gradient_checkpointing=True
