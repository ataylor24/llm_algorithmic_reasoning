import os
import argparse
import train_utils
from Model import GenerativeModel
from Dataset import AlgReasoningDataset, LlamaDataset
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    pipeline,
    logging,
    Trainer, 
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
from datasets import load_dataset
import wandb
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def evaluate(model, test_dataset, graph_size):
    import re
    INPUT_PATTERN = '\[INST\].*?\[/INST\]'
    
    score = 0

    inf_pipeline = pipeline(
    "text-generation",
    model=model.model,
    tokenizer=model.tokenizer,
    torch_dtype=torch.float16,
    device_map=config.device,
    )
    
    results = []
    pred_scores = []
    gold_scores = []
    for (prompt, gold_output) in test_dataset:
        pred_output = inf_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_new_tokens=20
        )[0]['generated_text']
        
        gold_output = re.findall(INPUT_PATTERN, gold_output, re.DOTALL)[0].strip()
        if pred_output.lower().find(gold_output.lower()) == -1:
            pred_score = 0
            gold_score = 1
        else:
            pred_score = 1
            gold_score = 1
        pred_scores.append(pred_score)
        gold_scores.append(gold_score)
        
        pred_scores.append(pred_score)
        gold_scores.append(gold_score)
        
        results.append((gold_output, pred_output, pred_score))
    
            
    return {
        "accuracy_score": accuracy_score(np.array(gold_scores), np.array(pred_scores)),
        "f1_score": f1_score(np.array(gold_scores), np.array(pred_scores)),
    }, results

def _load_data(graph_size):
    training_datapath = os.path.join(config.hint_level, "training" + "." + config.data_format)
    validation_datapath = os.path.join(config.hint_level, "validation" + "." + config.data_format)
    testing_datapath = os.path.join(config.hint_level, "testing" + "." + config.data_format)
    
    if config.model == "llama2":
        # dataloader = load_dataset("/local2/ataylor2/algorithmic_reasoning/bfs/graph_size_4/llm_data/llama2/with_hints")
        dataloader = load_dataset(f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.hint_level}")
    else:
        training_dataloader = DataLoader(AlgReasoningDataset(config, training_datapath))
        validation_dataloader = DataLoader(AlgReasoningDataset(config, validation_datapath))
        testing_dataloader = DataLoader(AlgReasoningDataset(config, testing_datapath))
        dataloader = (training_dataloader, validation_dataloader, testing_dataloader)
        
    return dataloader

def main(graph_size, inference_type):
    # start a new wandb run to track this script
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{config.algorithm}_{config.hint_level}_graph_size_{graph_size}"
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="algorithmic_reasoning",
        name= run_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.0002,
        "architecture": config.model,
        "algorithm": config.algorithm,
        "hint_level": config.hint_level,
        "epochs": 3,
        "batch_size":3,
        "ga_steps":1
        }
    )
    
    tokenizer_kwargs = {
        "padding": True,
        "truncation": True,
        "cache_dir": config.cache_dir,
        "use_fast": True,
        "token": None,
        # truncate from left, remove some input if too long
        "truncation_side": 'left',
        "padding_side": 'right',
        "model_max_length": config.model_max_length,
    }
    

    model = GenerativeModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, **tokenizer_kwargs)
    # Assign padding token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = _load_data(graph_size)
    
    bs=3       # batch size
    ga_steps=1  # gradient acc. steps
    epochs=3
    steps_per_epoch=len(dataset["train"])//(bs*ga_steps)

    training_args = TrainingArguments(
        output_dir="out",
        # per_device_train_batch_size=1,
        # per_device_eval_batch_size=1,
        # evaluation_strategy="steps",
        # logging_steps=1,
        # eval_steps=steps_per_epoch,  # eval and save once per epoch   
        # save_steps=steps_per_epoch,
        # gradient_accumulation_steps=ga_steps,
        # num_train_epochs=epochs,
        # lr_scheduler_type="cosine",
        # optim="paged_adamw_32bit",
        # learning_rate=0.0001,
        # group_by_length=True,
        fp16=True,
        # ddp_find_unused_parameters=False,
        # remove_unused_columns=False # needed for training with accelerate
    )
        
    if config.use_trl:
        if config.use_peft:
            # self.model = prepare_model_for_kbit_training(self.model)
            peft_config = LoraConfig(
                r=64, 
                lora_alpha=16, 
                target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
                lora_dropout=0.1, 
                bias="none", 
                modules_to_save = ["lm_head", "embed_tokens"],        # needed because we added new tokens to tokenizer/model
                task_type="CAUSAL_LM"
            )
        
        if inference_type == "answer_only":
            def formatting_prompts_func(example):
                output_texts = []
                for i in range(len(example['instruction'])): 
                    output_texts.append(f"{example['instruction'][i]} {example['input'][i]}\n### Reachable Nodes: [{example['output'][i]}]")
                return output_texts
            
            response_template = config.response_template
            
        elif inference_type == "show_reasoning":
            raise NotImplementedError(f"{inference_type} format not implemented yet!")
        else:
            raise NotImplementedError(f"{inference_type} is an invalid Inference format!")
         
        data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
        
        eval_dataset_dict = {
                'valid': dataset['validation'],
                'test': dataset['test']
            }
                   
        trainer = SFTTrainer(
            model=model.model,
            args=training_args,
            max_seq_length=config.model_max_length,
            train_dataset=dataset["train"] if config.do_train else None,
            eval_dataset=eval_dataset_dict if config.do_eval else None,
            formatting_func=formatting_prompts_func,
            # dataset_text_field="text",
            peft_config=peft_config if config.use_peft else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
        tokenized_train_data = []

        for example in dataset["train"]:
            tokenized_example = model.tokenize(example["text"])
            tokenized_train_data.append(tokenized_example)
        
        tokenized_val_data = []

        for example in dataset["validation"]:
            tokenized_example = model.tokenize(example["text"])
            tokenized_val_data.append(tokenized_example)
            
        trainer = Trainer(
            model=model.model,
            tokenizer=tokenizer,
            data_collator=model.collate,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_val_data,
            args=args,
        )
    
    trainer.train()
    
    metrics_dict, results = evaluate(model, train_utils.preprocess_test_data(dataset["test"]), graph_size)
    
    wandb.log(metrics_dict)
    
    print(metrics_dict)
    
    output_dir = train_utils.resolve_output_dirs(config, graph_size)
    train_utils.log_results(output_dir, run_name, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    parser.add_argument('graph_size', type=str, help='Path to the configuration JSON file')
    parser.add_argument('--inference_type', type=str, default="answer_only", help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    global config
    config = train_utils.load_json(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    main(args.graph_size, args.inference_type)