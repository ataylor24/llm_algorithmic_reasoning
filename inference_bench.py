import os, json, random, copy, re, time, pickle
import openai
from openai import OpenAI, AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from peft import PeftModel
import torch
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import copy
import gc
from statistics import mean, median
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
import argparse  # Added for argument parsing
import math

def get_lora_path(lora_base_path):
    lora_path = None
    for item in os.listdir(lora_base_path):
        if "checkpoint-" in item:
            lora_path = os.path.join(lora_base_path, item)
    return lora_path

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a given task.")
    parser.add_argument('--target_task', type=str, default='bfs', choices=['bfs', 'dfs', 'dijkstra', 'floyd_warshall', 'mst_prim'], help='Target task to perform')
    parser.add_argument('--reasoning_strategy', type=str, default='Int_Steps', help='Reasoning strategy to use')
    parser.add_argument('--graph_size', type=int, default=5, choices=[5,6,7,8,9,10,11,12,13,14,15,20,50], help='Size of the graph')
    parser.add_argument('--split_to_use', type=str, default='evaluate', help='Data split to use (train/test)')
    parser.add_argument('--mode', type=str, default='inference', help='Mode of operation (seq_gen/inference)')
    parser.add_argument('--inference_engine', type=str, default='hf', choices=['hf', 'openai', 'vllm'], help='Inference engine to use (hf/openai/vllm)')
    parser.add_argument('--llm_name', type=str, default='meta-llama/Meta-Llama-3-8B', choices=['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B', "mistralai/Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-v0.3", 'gpt-4o'], help='Name of the language model to use')
    parser.add_argument('--from_saved', action='store_true', help='Load from saved model or not')
    parser.add_argument('--chat_type', type=str, default='', help='allow inference on reruns')
    parser.add_argument('--verbose', type=bool, default=False, help='Primarily for debugging purposes.')
    return parser.parse_args()

args = parse_args()

# Use the parsed arguments
target_task = args.target_task
reasoning_strategy = args.reasoning_strategy
graph_size = args.graph_size
split_to_use = args.split_to_use
mode = args.mode
inference_engine = args.inference_engine
llm_name = args.llm_name
from_saved = args.from_saved
chat_type = args.chat_type
verbose = args.verbose

# If loading from a saved model
llm_path = llm_name
print("reasoning_strategy", reasoning_strategy)

if reasoning_strategy in ["IO_w_IS", "IO_w_IS_no_tf"]:
    base_reasoning_strategy = "IO"
elif reasoning_strategy in ["Int_Steps_w_IO"]:
    base_reasoning_strategy = "Int_Steps"
elif reasoning_strategy in ["IO_no_tf", "IO_nc_w_IS"]:
    base_reasoning_strategy = "IO_no_chat"
else:
    base_reasoning_strategy = reasoning_strategy

# Set the lora_base_path based on the base chat type
lora_base_path = f'/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/{chat_type}/{base_reasoning_strategy}'

lora_path = get_lora_path(lora_base_path)

if lora_path == None and not "gpt" in llm_name:
    lora_path = get_lora_path(lora_base_path.replace("local", "local2"))
    if lora_path == None and not "gpt" in llm_name:
        raise FileNotFoundError(f"There is no model checkpoint at: {lora_base_path}")
# lora_path = '/home/ubuntu/derek-240318/clinical-event-pred/alignment-handbook/data/llama3-8b-instruct-dpo-qlora-codes-diagnoses-full/checkpoint-4800'

save_path_parsed = 'data/mimic4/parsed'
hf_access_token = os.getenv('HF_TOKEN')
openai_engine = 'openai' # openai, azure
from_saved = True

if 'gpt' in llm_name:
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'hf':
    if '8x7b' in llm_name.lower():
        batch_size = 4
        batch_size_big = 40
    elif 'Mistral-7B' in llm_name:
        batch_size = 1
        batch_size_big = 1
    elif '7b' in llm_name.lower():
        batch_size = 1
        batch_size_big = 1
    else:
        batch_size = 1
        batch_size_big = 1
    batch_size = 1
    batch_size_big = 1
elif inference_engine == 'vllm':
    batch_size_big = 100

if mode == 'seq_gen' and 'Llama' not in llm_name:
    raise ValueError('Only Llama supports seq_gen mode')

save_name = llm_name.split('/')[-1] if '/' in llm_name else llm_name
if lora_path:
    lora_name = '-'.join(lora_path.split('/')[-2:]) if '/' in lora_path else lora_path
    if lora_name != '':
        lora_name = '_' + lora_name

# chat_type = "chat_gpt" if llm_name == 'gpt-4o' else "chat" + chat_suffix 

evaldata_save_path = f'/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/{chat_type}/{reasoning_strategy}/evaluation'
print("evaldata_save_path", evaldata_save_path)
batch_proc = False
char_per_token_est = 2 # -1, 4, 3, 2
token_count_lab = 500

# Prepare instructioin and system role for the selected task
prompt_system_role = ''
if not target_task in ['bfs', 'floyd_warshall', 'dfs', 'dijkstra', 'mst_prim']:
    assert False, 'Not implemented for other tasks'

if 'gpt' in llm_name:
    if openai_engine == 'azure':
        print(f'Using Azure AI {llm_name}')
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-02-01",
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=30
        )
        if 'gpt-4' in llm_name:
            azure_deployment_name='gpt4-turbo'
        else:
            azure_deployment_name='medical'
    else:
        print(f'Using Open AI {llm_name}')
        # Prepare Open AI API
        client = OpenAI(timeout=30)
else:
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    
    model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16, device_map="auto")
    default_generation_config = model.generation_config
    default_max_length = model.config.max_position_embeddings
    default_max_length_output = 5 * graph_size + 7 #max(model.generation_config.max_length, 500)
    tokenizer.chat_template = """
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ eos_token }}
{% endif %}
{% endfor %}
""" 
    if inference_engine == 'hf':
        print(f'Using Huggingface {llm_path}')
        if batch_proc:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            if lora_path == '':
                del model
                gc.collect()
                torch.cuda.empty_cache()
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=llm_path,
                    return_full_text=False,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
            else:
                model = PeftModel.from_pretrained(
                    model,
                    lora_path,
                    torch_dtype=torch.float16,
                )
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model,
                    return_full_text=False,
                    tokenizer=tokenizer,
                    framework="pt",
                )
            pipeline.tokenizer.pad_token_id = pipeline.model.config.eos_token_id

# Set max token length for each model
if llm_name == 'gpt-4-turbo':
    max_token_length = 128000
elif llm_name == 'gpt-3.5-turbo':
    max_token_length = 16385
elif llm_name == 'gpt-4o':
    max_token_length = 128000
else:
    assert default_max_length <= 65536, f'the max_token_length {default_max_length} is probably wrong, please double check'
    max_token_length = default_max_length - 70

# 
prompt_basic_length_char = len(prompt_system_role) 
left_char_count = max_token_length * char_per_token_est - prompt_basic_length_char


def model_specific_prompt(datapoint):
    if 'gpt' in llm_name.lower():
        return datapoint[0]
    if 'llama' in llm_name.lower() or 'alpaca' in llm_name.lower() or 'mistral' in llm_name.lower():
        # not having <s> at the beginning, since it will be added by tokenizer automatically
        return tokenizer.apply_chat_template(datapoint, tokenize=False, add_generation_prompt=False)
    
def generate_partial_examples(data):
    examples = []
    ground_truths = []
    intermediate_prompt = None
    
    if 'gpt' in llm_name.lower() and reasoning_strategy != "Int_Steps":
        for i in range(len(data)):
            if i == 0:
                partial_example = data[i]["content"]
                
                examples.append(partial_example)
            elif i % 2 == 0:
                partial_example = "".join([dp["content"] for dp in data[:i + 1]]) + "\n"
                
                examples.append(partial_example)
            else:
                ground_truths.append(data[i]["content"])
    elif 'gpt' in llm_name.lower() and reasoning_strategy == "Int_Steps":
        for i in range(len(data)):
            if i == 0:
                partial_example = [data[0]["content"]]
                examples.append(partial_example)
            elif i % 2 == 0:
                partial_example = "".join([dp["content"] for dp in data[:i + 1]])
                examples.append(partial_example)
            else:
                ground_truths.append(data[i]["content"])
    elif "no_tf" in reasoning_strategy:
        for i in range(len(data)):
            if i == 0:
                partial_example = [data[0]]
                partial_example.append({'role': 'assistant', 'content': 'assistant:'})
                examples = partial_example
            elif i % 2 == 0 and intermediate_prompt == None:
                intermediate_prompt= data[i]
            else:
                ground_truths.append(data[i])
    else:
        for i in range(len(data)):
            if i == 0:
                partial_example = [data[0]]
                partial_example.append({'role': 'assistant', 'content': 'assistant:'})
                examples.append(partial_example)
            elif i % 2 == 0:
                partial_example = data[:i + 1]
                partial_example.append({'role': 'assistant', 'content': 'assistant:'})
                examples.append(partial_example)
            else:
                ground_truths.append(data[i])
   
    return examples, ground_truths, intermediate_prompt

def inference(dps, verbose=False):
    responses = []
    outputs = []
    dps_is_list = True
    if not isinstance(dps, list):
        dps = [dps]
        dps_is_list = False
    if 'gpt' in llm_name:
        for dp in dps:
            response = None
            output = []

            try_count = 0
            success_gen_flag = False
            
            for messages in dp["messages"]:
                partial_messages, gts, _ = generate_partial_examples(messages)
                for message, gt in zip(partial_messages,gts):
            
                    while success_gen_flag is False and try_count < 3:
                        try_count += 1
                        try:
                            # first time, use full ideal message
                            message_formatted = model_specific_prompt(message)
                            # if 'gpt-4' in llm_name:
                            #     time.sleep(2)
                          
                            
                            messages_this_call = [
                                                    {"role": "user", "content": message_formatted},
                                            
                                                ]
                            if openai_engine == 'azure':
                                response = client.chat.completions.create(
                                    model=azure_deployment_name,
                                    messages=messages_this_call
                                )
                            else:
                                response = client.chat.completions.create(
                                    model=llm_name,
                                    messages=messages_this_call
                                )
                            for choice in response.choices:
                                output.append(choice.message.content)
                            success_gen_flag = True
                        except openai.APIError as e:
                            #Handle API error here, e.g. retry or log
                            print(f"OpenAI API returned an API Error: {e}")
                            success_gen_flag = False
                            pass
                        except openai.APIConnectionError as e:
                            #Handle connection error here
                            print(f"Failed to connect to OpenAI API: {e}")
                            success_gen_flag = False
                            pass
                        except openai.RateLimitError as e:
                            #Handle rate limit error (we recommend using exponential backoff)
                            print(f"OpenAI API request exceeded rate limit: {e}")
                            success_gen_flag = False
                            pass
                        except UserWarning as e:
                            print(f"UserWarning: {e}")
                            if 'Input length of input_ids is' in e:
                                success_gen_flag = False
                            pass
                    outputs.append({
                        "traj_id": dp["traj_id"][0],
                        "message": message,
                        "ground_truth": gt,
                        "pred": output,
                    })
               
    elif inference_engine == 'hf':
        
        outputs = []
        if batch_proc:
            chats = [dp['messages'] for dp in dps] 
            inputs = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
            
            output_sequences = model.generate(
                **inputs,
                do_sample=True,
                # top_k=10, # might lead to RuntimeError: probability tensor contains either `inf`, `nan` or element < 0
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                max_length=default_max_length_output,
            )
            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif "no_tf" in reasoning_strategy:
            for dp in dps:
                for messages in dp['messages']:
                # chats = [dp['messages'] for dp in dps] 
                    # if len(chats[0]) == 0:
                    #     return outputs
                    problem_statment, gts, intermediate_prompt = generate_partial_examples(messages)
                    for i, gt in enumerate(gts):
                        if i == 0:
                            message = problem_statment
                        else:
                            message = copy.copy(message)
                            message.extend([intermediate_prompt, {'role': 'assistant', 'content': 'assistant:'}])
                        try:
                            outputs_batch = pipeline(
                                model_specific_prompt(message),
                                do_sample=True,
                                num_return_sequences=1,
                                # batch_size=batch_size,
                                eos_token_id=tokenizer.eos_token_id,
                                max_new_tokens=default_max_length_output,
                            )
                            # for outputs_single in outputs_batch:
                            #     outputs.append(outputs_single['generated_text'])
                            
                        except Exception as e:
                            print(f"Failed to process {[i for i, dp in enumerate(dps)]}: {e}")
                            
                            message[-1]["content"] = 'assistant: []'
                            
                            outputs.append({
                            "message": message,
                            "ground_truth": gt,
                            "pred": []
                            })
                            
                            continue
                        
                        message[-1]["content"] = f'assistant:{outputs_batch[0]["generated_text"]}'
                    
                        outputs.append({
                            "traj_id": dp["traj_id"][0],
                            "message": message,
                            "ground_truth": gt,
                            "pred": outputs_batch[0]
                        })
                        
        else:
            for dp in dps:
                for messages in dp['messages']:
                # chats = [dp['messages'] for dp in dps] 
                    # if len(chats[0]) == 0:
                    #     return outputs
                    messages, gts, _ = generate_partial_examples(messages)#chats[0][0])
                    for message, gt in zip(messages, gts):
                        try:
                            outputs_batch = pipeline(
                                model_specific_prompt(message),
                                do_sample=True,
                                num_return_sequences=1,
                                # batch_size=batch_size,
                                eos_token_id=tokenizer.eos_token_id,
                                max_new_tokens=default_max_length_output,
                            )
                            # for outputs_single in outputs_batch:
                            #     outputs.append(outputs_single['generated_text'])
                            
                        except Exception as e:
                            print(f"Failed to process {[i for i, dp in enumerate(dps)]}: {e}")
                            outputs.append({
                            "message": message,
                            "ground_truth": gt,
                            "pred": []
                        })
                            continue
                        
                        outputs.append({
                            "traj_id": dp["traj_id"][0],
                            "message": message,
                            "ground_truth": gt,
                            "pred": outputs_batch[0]
                        })
                
    # if not dps_is_list:
    #     if 'gpt' in llm_name:
    #         responses = responses[0]
    #     outputs = outputs[0]
    return outputs


if from_saved and mode == 'inference' and os.path.exists(evaldata_save_path):
    # evaldata = pickle.load(open(evaldata_save_path, 'rb'))
    evaldata = load_from_disk(evaldata_save_path)
    print(f'-> Using cached evaldata saved at {evaldata_save_path}')
else:
    with open(f'data/mimic4/{target_task}/{split_to_use}.pkl', 'rb') as f:
        evaldata = pickle.load(f)

    gendata = []

    cut_count_user_msg_end = 0
    cut_count_user_msg_note = 0
    cut_rates = []
    if 'gpt' not in llm_name.lower():
        # For open-source models, we can cut the input directly here
        prompt_basic = prompt_system_role
        left_token_count = max_token_length - len(tokenizer(prompt_basic)['input_ids']) - 30
        print(f'left_token_count: {left_token_count}')

    for dp_i, dp in enumerate(tqdm(evaldata, desc=f"{target_task}, {llm_name}, gen data")):
       

        # For GPT models, the full input will be generated dynamically when provide the input
        evaldata[dp_i]['input_raw'] = [
            dp
        ]

        if 'gpt' not in llm_name.lower():
            # Truncate the input message to fit the model input length
            # For open-source models, we can cut the input directly here
            
            
            # Option 1: use our function to fill in prompt template -> not up to date
            # evaldata[dp_i]['input'] = model_specific_prompt(prompt_system_role, prompt_task_instruction, user_message_cut, prompt_task_instruction_end)
            # Option 2: use huggingface chat template
            if 'mistral' in llm_name.lower():
                messages_this = [
                    {"role": "user", "content": prompt_system_role + '\n'}
                ]
            else:
                messages_this = [
                    {"role": "system", "content": prompt_system_role + '\n'},
                    #{"role": "user", "content": user_message_cut + '\n'}
                ]
            evaldata[dp_i]['input'] = tokenizer.apply_chat_template(messages_this, tokenize=False, add_generation_prompt=True) 

            # Prepare target sequence used for seq2seq model training
            
            gt_codes = [item for item in dp['Assistant']]
            evaldata[dp_i]['target_gold'] = "\n".join(gt_codes)
            

            gen_dp = {
                'id': dp['hadm_id'],
                'text': evaldata[dp_i]['input'] + evaldata[dp_i]['target_gold'],
                'input': evaldata[dp_i]['input'],
                'target_gold': evaldata[dp_i]['target_gold'],
                'eval_gold': evaldata[dp_i]['target_gold'],
            }
            gendata.append(gen_dp)

    if 'gpt' not in llm_name.lower():
        print(f'{cut_count_user_msg_note}/{len(evaldata)} input sequence cut discharge + radiology note')
        print(f'{cut_count_user_msg_end}/{len(evaldata)} input sequence cut the end of patient record')
        print(f'cut rates mean: {mean(cut_rates)}, medium: {median(cut_rates)}, min: {min(cut_rates)}, max: {max(cut_rates)}')
        # Statistics of input length and target_gold of the gen dataset
        lengths_i = [len(tokenizer(dp['input'])['input_ids'][1:]) for dp in gendata]
        lengths_o = [len(tokenizer(dp['target_gold'])['input_ids'][1:]) for dp in gendata]
        print('Statistics of number of tokens in input and target_gold of the gen dataset:')
        for list_this in [lengths_i, lengths_o]:
            print(f"Mean: {mean(list_this)}")
            print(f"Median: {median(list_this)}")
            print(f"Min: {min(list_this)}")
            print(f"Max: {max(list_this)}")
        if mode == 'seq_gen':
            with open(f'data/mimic4/{target_task}/{split_to_use}_gen.json', 'w', encoding='utf-8') as f:
                json.dump(gendata, f, indent=4)
            print('-> gen io saved.')

    pickle.dump(evaldata, open(evaldata_save_path, 'wb'))

# default_max_length_output = graph_size
if inference_engine == 'hf':
    avg_length_output = []
    for dp in evaldata:
        for message in dp["messages"][1:]:
            message_tokens = tokenizer(message["content"])
            avg_length_output.append(len(message_tokens["input_ids"]))
            # if len(message_tokens["input_ids"]) > default_max_length_output:
            #     default_max_length_output = len(message_tokens["input_ids"])

    default_max_length_output = min(sorted(avg_length_output)[:math.floor(0.90*len(avg_length_output)) + 1][-1], 8192)
    print("default_max_length_output:", default_max_length_output)
    
if mode == 'inference':
    results = copy.deepcopy(evaldata)
    # if os.path.exists(f'/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/chat/{reasoning_strategy}/{save_name}_inference.json'):
    #     with open(f'/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/chat/{reasoning_strategy}/{save_name}_inference.json', 'r') as f:
    #         results_2 = json.load(f)
    #     # generation for this dp has been done, skip
    #     # generated content is long enough, otherwise not skip
    #     results_2 = [dp for dp in results_2]# if len(dp['output']) > 0 and len(dp['output'][0]) > 100]
    #     # ids_done = [result['hadm_id'] for result in results_2]
    # else:
    #     results_2 = []
    results_2 = []
    # skip inference for the data instances that already have generated output
    # results = [dp for dp in results if dp['hadm_id'] not in ids_done]

    # if not os.path.exists(f'data/mimic4/{target_task}_output'):
    #     os.makedirs(f'data/mimic4/{target_task}_output')
  
    global_verbose_flag = True

    for batch_i in tqdm(range(len(results) // batch_size_big + 1), desc=f"{target_task}, {llm_name}, inference"):
        input_dps = results[batch_i * batch_size_big: (batch_i + 1) * batch_size_big]
        outputs = inference(input_dps, verbose=global_verbose_flag)
       
        # for in_batch_i, result in enumerate(input_dps):
        #     result_new = {
        #         # 'hadm_id': result['hadm_id'],
        #         'output': outputs[in_batch_i]
        #     }
        #     results_2.append(result_new)
        #     # ids_done.append(result['hadm_id'])
        #     global_verbose_flag = False
        for output in outputs:
            results_2.append(output)
            
            if verbose:
                if not isinstance(output, str):
                    if inference_engine == 'hf':
                        print(output["ground_truth"]["content"], output["pred"]["generated_text"])
                    else:
                        print(output["pred"])
        
    print("Dumping results:", f"/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/{chat_type}/{reasoning_strategy}/{save_name}_inference.json")    
    with open(f"/local/ataylor2/algorithmic_reasoning/{target_task}/graph_size_{graph_size}/llm_data/{chat_type}/{reasoning_strategy}/{save_name}_inference.json", 'w', encoding='utf-8') as f:
        json.dump(results_2, f, indent=4)