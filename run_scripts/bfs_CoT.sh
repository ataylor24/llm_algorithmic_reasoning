export CUDA_VISIBLE_DEVICES=4
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN=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
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 ../scripts/run_sft.py ../recipes/algorithmic_reasoning/sft/config_qlora.yaml --load_in_4bit=true
