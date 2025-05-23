import gc
import os
import torch
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Optional, Dict
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import KTOConfig, KTOTrainer, is_xpu_available
from datetime import datetime
from datasets import load_from_disk
import pandas as pd
import random

# Config
input_data_dir = "./data/"
base_model_name = "Qwen/Qwen2.5-7B"
padding_side_req = "right" #by default mistral uses left, but trl has overflow issues with left padding.
output_dir = "./kto_0.03"
max_seq_length = 2048
max_prompt_length = 1024
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
# bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )

# Base template
BASE_TEMPLATE = '''<bos><im_start>user\n {prompt} <im_end>\n<im_start>assistant\n'''
RESPONSE_TEMPLATE = '''{response}<im_end>'''

def filter_by_length(example, tokenizer):
    len_prompt = len(tokenizer(example["prompt"], padding=False, truncation=False)["input_ids"])
    len_completion = len(tokenizer(example["completion"], padding=False, truncation=False)["input_ids"])

    return True if (len_prompt <= max_prompt_length and len_prompt + len_completion <= max_seq_length) and (len_prompt != 0) and (len_completion != 0) else False

def apply_formatting(example, tokenizer):
    prompt = BASE_TEMPLATE.format(prompt=example["prompt"])
    chosen_completion = example["chosen"][-1]["content"]
    chosen_label = True
    chosen_rating = (example["score_chosen"]-1)/9
    rejected_completion = example["rejected"][-1]["content"]
    rejected_label = False
    rejected_rating = (example["score_rejected"]-1)/9
    
    return [{"prompt": prompt, "completion": chosen_completion, "label": chosen_label, "average_rating": chosen_rating}, 
            {"prompt": prompt, "completion": rejected_completion, "label": rejected_label, "average_rating": rejected_rating}]

'''
def load_and_format_datasets(input_data_dir, tokenizer):
    dataset_dict_clean = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    dataset_dict_clean_pd = pd.DataFrame(dataset_dict_clean['train'])  # assuming 'train' split

    modified_dataset = []
    for index, row in dataset_dict_clean_pd.iterrows():
        modified_row = apply_formatting(row)
        modified_dataset.extend(modified_row)
    random.seed(42)
    random.shuffle(modified_dataset)

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=modified_dataset))
    #train_dataset = dataset_dict_clean['train'].shuffle(seed=42).map(lambda example: apply_formatting(example))
    train_dataset = train_dataset.filter(lambda example: filter_by_length(example, tokenizer))
    
    return train_dataset
'''

def apply_formatting_helpsteer(example1, example2, tokenizer):
    prompt = BASE_TEMPLATE.format(prompt=example1["prompt"])
    response1 = example1["response"]
    response2 = example2["response"]
    score1 = 0.65 * example1["helpfulness"] + 0.8 * example1["correctness"] + 0.45 * example1["coherence"]
    score2 = 0.65 * example2["helpfulness"] + 0.8 * example2["correctness"] + 0.45 * example2["coherence"]
    max_score = 0.65 * 4 + 0.8 * 4 + 0.45 * 4
    score1_norm = score1 / max_score
    score2_norm = score2 / max_score

    if score1_norm > score2_norm:
        return [{"prompt": prompt, "completion": response1, "label": True, "average_rating": score1_norm}, 
                {"prompt": prompt, "completion": response2, "label": False, "average_rating": score2_norm}]
    else:
        return [{"prompt": prompt, "completion": response2, "label": True, "average_rating": score2_norm}, 
                {"prompt": prompt, "completion": response1, "label": False, "average_rating": score1_norm}]

def load_and_format_datasets(input_data_dir, tokenizer):
    dataset_dict_clean = load_dataset("nvidia/HelpSteer2")
    dataset_dict = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    #train
    dataset_dict_clean_pd = pd.DataFrame(dataset_dict_clean['train'])  # assuming 'train' split
    modified_dataset = []
    for index in range(0, len(dataset_dict_clean_pd), 2):
        row1 = dataset_dict_clean_pd.iloc[index]
        row2 = dataset_dict_clean_pd.iloc[index + 1] if index + 1 < len(dataset_dict_clean_pd) else None

        modified_row = apply_formatting_helpsteer(row1, row2, tokenizer)
        modified_dataset.extend(modified_row)

    random.seed(42)
    random.shuffle(modified_dataset)

    train_dataset = Dataset.from_pandas(pd.DataFrame(data=modified_dataset))
    train_dataset = train_dataset.filter(lambda example: filter_by_length(example, tokenizer))

    # dev
    dataset_dict_pd = pd.DataFrame(dataset_dict['test_prefs'])  # assuming 'train' split
    modified_dataset = []
    for index in range(0, len(dataset_dict_pd)):
        row = dataset_dict_pd.iloc[index]
        modified_row = apply_formatting(row, tokenizer)
        modified_dataset.extend(modified_row)

    dev_dataset = Dataset.from_pandas(pd.DataFrame(data=modified_dataset))
    dev_dataset = dev_dataset.filter(lambda example: filter_by_length(example, tokenizer))

    return train_dataset, dev_dataset

def initialize_tokenizer_and_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        #quantization_config=bnb_config,
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
        device_map = (
        {"": f"xpu:{Accelerator().local_process_index}"}
        if is_xpu_available()
        else {"": Accelerator().local_process_index}
    ))
    base_model.config.use_cache = False #use_cache=False if gradient_checkpointing else True
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, 
                                              add_bos_token=False, 
                                              add_eos_token=False)
    tokenizer.pad_token = tokenizer.eos_token #If you set the eos_token as a padding token, the tokenizer will set the eos_token attention mask as “0”. The model will tend to ignore the eos_token and might over-generate tokens, which is not ideal for a down-streaming task. A more suitable option will be unk_token , because of its rarity, it will barely degrade the model’s performance even if we set its attention mask to “0” (i.e., set it as a padding token)
    new_specials = {"bos_token": "<bos>"}  
    num_added_toks = tokenizer.add_special_tokens(new_specials)
    tokenizer.padding_side = padding_side_req

    return tokenizer, base_model


def main():
    tokenizer, policy_model = initialize_tokenizer_and_model()
    train_dataset, dev_dataset = load_and_format_datasets(input_data_dir, tokenizer)
    # training config
    training_args = KTOConfig(
        output_dir=output_dir,
        warmup_ratio=0.1,
        per_device_train_batch_size=1,
        num_train_epochs=2,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5.0e-6,
        logging_steps=50,
        lr_scheduler_type="cosine",
        optim="paged_adamw_32bit",
        bf16=True,
        do_eval=False,
        #save_strategy="epoch",
        save_strategy="steps",
        save_steps=1000,
        remove_unused_columns=False,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        run_name=f"logs-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",
        beta=0.03,
        desirable_weight=1.0,
        undesirable_weight=1.0,
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
    )
    # ref model would be a clone of policy model if not specificed
    # trl takes care of its initialization

    kto_trainer = KTOTrainer(
        model=policy_model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
        args=training_args, 
    )

    kto_trainer.train()
    #kto_trainer.save_model(os.path.join(output_dir, "save_model"))
    # Save artifacts
    kto_trainer.model.save_pretrained("final_checkpoint")
    tokenizer.save_pretrained("final_checkpoint")
    # Flush memory
    del kto_trainer, policy_model
    gc.collect()
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()