export CUDA_VISIBLE_DEVICES=4
export HF_HOME=/local2/ataylor2/algorithmic_reasoning/cache
export HF_TOKEN=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
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file /home/ataylor2/algorithmic_reasoning/proj_baseline/recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port=45803 /home/ataylor2/algorithmic_reasoning/proj_baseline/scripts/run_sft.py /local2/ataylor2/algorithmic_reasoning/floyd_warshall/graph_size_8/llm_data/chat/Int_Steps/config_qlora.yaml --load_in_4bit=true