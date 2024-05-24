import os
import argparse
import train_utils
from Model import GenerativeModel
from Dataset import AlgReasoningDataset, LlamaDataset
import torch
import re
from datetime import datetime
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import (
    TrainingArguments,
    pipeline,
    Trainer, 
    EarlyStoppingCallback,
    AutoTokenizer,
    default_data_collator
)
from datasets import load_dataset, DatasetDict, Dataset
import wandb
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import load_metric

exact_match_metric = load_metric("exact_match")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def compute_metric(eval_preds):
    predictions, references = eval_preds
    return exact_match_metric.compute(predictions=predictions, references=references)

def evaluate(model, tokenizer, test_dataset, run_name, graph_size):
    gold_output_pattern = r"Reachable Nodes:\s*(\[.*?\])"
    semi_gold_output_pattern = r"\s*(\[.*?\])"
    
    results = []
    accuracy = []
    partial_accuracy = []
    avg_f1 = []
    avg_partial_f1 = []
    f1 = [[], []]
    partial_f1 = [[], []]

    inf_pipeline = pipeline(
    "text-generation",
    model=model.model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    return_full_text=False
    )

    for (prompt, gold_output) in test_dataset:
        pred_output = inf_pipeline(
            prompt,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            max_new_tokens=64
        )[0]['generated_text']
        
        gold_extracted = re.search(gold_output_pattern, gold_output, re.IGNORECASE)
        pred_extracted = re.search(gold_output_pattern, pred_output, re.IGNORECASE)
        
        partial_gold_extracted = re.search(semi_gold_output_pattern, gold_output, re.IGNORECASE)
        partial_pred_extracted = re.search(semi_gold_output_pattern, pred_output, re.IGNORECASE)
        
        pred = train_utils.unpack_literal(pred_extracted.group(1)) if pred_extracted != None else None
        gold = train_utils.unpack_literal(gold_extracted.group(1))
        
        partial_pred = train_utils.unpack_literal(partial_pred_extracted.group(1)) if partial_pred_extracted != None else None
        partial_gold = train_utils.unpack_literal(partial_gold_extracted.group(1))
        
        if pred != None:
            pred_f1 = train_utils.list_to_indicator(pred, graph_size)  
            
        else:
            pred_f1 = [-1] * graph_size
            
        if partial_pred != None:
            partial_pred_f1 = train_utils.list_to_indicator(partial_pred, graph_size)  
            
        else:
            partial_pred_f1 = [-1] * graph_size
            
        gold_f1 = train_utils.list_to_indicator(gold, graph_size)
        partial_gold_f1 = train_utils.list_to_indicator(partial_gold, graph_size)
        
        print(pred_output, gold_output)
        
        accuracy.append(int(pred == gold))
        avg_f1.append(f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"))
        f1[0].extend(gold_f1)
        f1[1].extend(pred_f1)
        
        partial_accuracy.append(int(partial_pred == partial_gold))
        avg_partial_f1.append(f1_score(np.array(partial_gold_f1), np.array(partial_pred_f1), average="micro"))
        partial_f1[0].extend(partial_gold_f1)
        partial_f1[1].extend(partial_pred_f1)

                
        results.append({
            "gold_output": gold_output,
            "pred_output": pred_output,
            "accuracy (Exact Match)": gold == pred,
            "f1 (Exact Match)": f1_score(np.array(gold_f1), np.array(pred_f1), average="micro"),
            "accuracy (List Only)": partial_pred == partial_gold,
            "f1 (List Only)": f1_score(np.array(partial_gold_f1), np.array(partial_pred_f1), average="micro"),
            })
    
            
    metrics_dict = {
        "accuracy_score": sum(accuracy)/len(accuracy),
        "avg_f1_score": sum(avg_f1)/len(avg_f1),
        "f1_confusion_matrix": confusion_matrix(np.array(f1[0]), np.array(f1[1])).tolist(),
        "partial_accuracy_score": sum(partial_accuracy)/len(partial_accuracy),
        "partial_avg_f1_score": sum(avg_partial_f1)/len(avg_partial_f1),
        "partial_f1_confusion_matrix": confusion_matrix(np.array(partial_f1[0]), np.array(partial_f1[1])).tolist()
    }
    
    results.insert(0, metrics_dict)
    
    return metrics_dict, results

def _load_data(graph_size):
    base_path = f"{config.data_path}/graph_size_{graph_size}/{config.dataset_type}/{config.model}/{config.hint_level}"
    training_datapath = f"{base_path}/training.{config.data_format}"
    validation_datapath = f"{base_path}/validation.{config.data_format}"
    testing_datapath = f"{base_path}/testing.{config.data_format}"
    
    if config.model in train_utils.ACCEPTED_MODELS:
        training_dataloader = load_dataset(base_path, "train", download_mode='force_redownload')
        validation_dataloader = load_dataset(base_path, "validation", download_mode='force_redownload')
        testing_dataloader = load_dataset(base_path, "test", download_mode='force_redownload')
        
        dataloader = (training_dataloader, validation_dataloader, testing_dataloader)
    else:
        training_dataloader = DataLoader(AlgReasoningDataset(config, training_datapath))
        validation_dataloader = DataLoader(AlgReasoningDataset(config, validation_datapath))
        testing_dataloader = DataLoader(AlgReasoningDataset(config, testing_datapath))
        dataloader = (training_dataloader, validation_dataloader, testing_dataloader)
        
    return dataloader

def main(graph_size, inference_type):
    
    output_dir = train_utils.resolve_output_dirs(config, graph_size)
    
    # start a new wandb run to track this script
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{config.model}_{config.algorithm}_{config.hint_level}_graph_size_{graph_size}"
    
    bs=1       # batch size
    ga_steps=1 # gradient acc. steps
    epochs=6 if graph_size > 4 else (500 if graph_size == 3 else 48)
    steps_per_epoch= 1/20# len(dataset["train"]) * epochs // 3
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="algorithmic_reasoning",
        name= run_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.0001,
        "lr_scheduler_type": "cosine",
        "optim": "paged_adamw_32bit",
        "architecture": config.model,
        "algorithm": config.algorithm,
        "hint_level": config.hint_level,
        "epochs": epochs,
        "batch_size":bs,
        "ga_steps":ga_steps
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
        "max_seq_length":1024
    }
    

    model = GenerativeModel(config)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path, **tokenizer_kwargs)
    # Assign padding token for tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
        
    dataset = _load_data(graph_size)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, 'model_checkpoints'),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        evaluation_strategy="epoch" if graph_size > 4 else "steps",
        save_strategy="epoch" if graph_size > 4 else "steps",
        logging_steps= 10,
        gradient_accumulation_steps=ga_steps,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        learning_rate=0.0001,
        group_by_length=True,
        do_predict=True,
        load_best_model_at_end=True,
        # fp16=True,
        # ddp_find_unused_parameters=False,
        # remove_unused_columns=False # needed for training with accelerate
    )
    if graph_size <= 4:
        training_args.eval_steps = steps_per_epoch  
        training_args.save_steps = steps_per_epoch
    
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
        
    if config.use_trl:
        
        if inference_type == "answer_only":
            def formatting_prompts_func(example):
                output_texts = []
                for i in range(len(example['instruction'])): 
                    if 'hints' in example:
                        output_texts.append(f"{example['instruction'][i]} {example['input'][i]}\n{example['output'][i]}")
                    else:
                        output_texts.append(f"{example['instruction'][i]} {example['input'][i]}\n{example['output'][i]}")
                return output_texts
            
            # response_template = config.response_template 
            
        elif inference_type == "show_reasoning":
            raise NotImplementedError(f"{inference_type} format not implemented yet!")
        else:
            raise NotImplementedError(f"{inference_type} is an invalid Inference format!")
         
        data_collator = DataCollatorForCompletionOnlyLM(config.response_template , tokenizer=tokenizer)
        
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
            data_collator=data_collator,                  # evaluation dataset
            compute_metrics=compute_metric,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    else:
        # tokenized_train_data = []

        # for example in dataset["train"]:
        #     tokenized_example = model.tokenize(example["text"])
        #     tokenized_train_data.append(tokenized_example)
        
        # tokenized_val_data = []

        # for example in dataset["validation"]:
        #     tokenized_example = model.tokenize(example["text"])
        #     tokenized_val_data.append(tokenized_example)
      
        data_collator = DataCollatorForCompletionOnlyLM(config.response_template, tokenizer=tokenizer)
        # tokenized_train_data = dataset[0]["train"].map(tokenize_chat)
        
        def formatting_func(example):
            return example
        
        tokenized_train_data = []
        for i in dataset[0]["train"][0]:
            tokenized_train_data.append(tokenizer.apply_chat_template(dataset[0]["train"][0][i], tokenize=False, add_generation_prompt=False))
        
        tokenized_validation_data = []
        for i in dataset[1]["train"][0]:
            tokenized_validation_data.append(tokenizer.apply_chat_template(dataset[1]["train"][0][i], tokenize=False, add_generation_prompt=False))
   
        # Convert the tokenized data to a Dataset object
        train_dataset = Dataset.from_dict({"formatted_chat": tokenized_train_data})
        validation_dataset = Dataset.from_dict({"formatted_chat": tokenized_validation_data})
        # tokenized_train_data = dataset[0]["train"].map(lambda x: {"formatted_chat": tokenizer.apply_chat_template([x[str(i)] for i in range(len(x))], tokenize=True, add_generation_prompt=False)})
        
        # tokenized_val_data = dataset[1]["validation"].map(lambda x: {"formatted_chat": tokenizer.apply_chat_template([x[str(i)] for i in range(len(x))], tokenize=True, add_generation_prompt=False)})
            
        trainer = SFTTrainer(
            model=model.model,
            args=training_args,
            max_seq_length=config.model_max_length,
            train_dataset=train_dataset if config.do_train else None,
            eval_dataset=validation_dataset if config.do_eval else None,
            peft_config=peft_config if config.use_peft else None,
            dataset_text_field="formatted_chat",
            # formatting_func=formatting_func,
            tokenizer=tokenizer,
            # data_collator=default_data_collator,                  # evaluation dataset
            compute_metrics=compute_metric,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    trainer.train()
    
    # metrics_dict, results = evaluate(model, tokenizer, train_utils.preprocess_test_data(dataset["test"]), run_name, graph_size)
    
    # wandb.log(metrics_dict)
    
    # train_utils.print_metrics_dict(metrics_dict)
   
    
    # train_utils.log_results(output_dir, run_name, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    parser.add_argument('graph_size', type=int, help='Path to the configuration JSON file')
    parser.add_argument('--inference_type', type=str, default="answer_only", help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    global config
    config = train_utils.load_json(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    main(args.graph_size, args.inference_type)