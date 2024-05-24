export CUDA_VISIBLE_DEVICES=6
export TRANSFORMERS_CACHE=/local2/ataylor2/algorithmic_reasoning/cache
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file ../recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 ../scripts/run_sft.py ../recipes/algorithmic_reasoning/sft/config_qlora.yaml --load_in_4bit=true
