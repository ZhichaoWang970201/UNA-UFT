import gc
import os
import torch
import bitsandbytes as bnb
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
from accelerate import Accelerator
from datasets import load_dataset, Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from trl import DPOTrainer, is_xpu_available
from datetime import datetime
from datasets import load_from_disk
import pandas as pd
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
import random
import pandas as pd

class UNA_BCE(DPOTrainer):
    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a DPO specific dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.model, padded_batch)
            else:
                (
                    reference_chosen_logps,
                    reference_rejected_logps,
                    _,
                    _,
                    _,
                    _,
                ) = self.concatenated_forward(self.ref_model, padded_batch)

        return reference_chosen_logps, reference_rejected_logps
    
    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """Tokenize a single row from a DPO specific dataset.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
        in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        batch["chosen_rating"] = feature["chosen_rating"]
        batch["rejected_rating"] = feature["rejected_rating"]

        if not self.is_encoder_decoder:
            # Check issues below for more details
            #  1. https://github.com/huggingface/trl/issues/907
            #  2. https://github.com/EleutherAI/lm-evaluation-harness/pull/531#issuecomment-1595586257
            #  3. https://github.com/LianjiaTech/BELLE/issues/337

            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            # Last prompt token might get merged by tokenizer and
            # it should not be included for generation if that happens
            prompt_len_input_ids = len(prompt_tokens["prompt_input_ids"])

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Make sure prompts only have one different token at most an
            # and length only differs by 1 at most
            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            # add BOS token to head of prompt
            prompt_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + prompt_tokens["prompt_input_ids"]
            chosen_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + chosen_tokens["prompt_input_ids"]
            rejected_tokens["prompt_input_ids"] = [self.tokenizer.bos_token_id] + rejected_tokens["prompt_input_ids"]

            prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            # add EOS token to end of answer
            chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            chosen_tokens["attention_mask"].append(1)

            rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
            rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            # if combined sequence is too long, truncate the prompt
            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            # if that's still too long, truncate the response
            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            # Create labels
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_rating = batch['chosen_rating'][0]
        rejected_rating = batch['rejected_rating'][0]

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_rating, rejected_rating)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        chosen_rating: torch.FloatTensor,
        rejected_rating: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities.

        Args:
            policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
            policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
            reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
            reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

        Returns:
            A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
            The losses tensor contains the DPO loss for each example in the batch.
            The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
        """

        if self.reference_free:
            chosen_logps = self.beta * policy_chosen_logps
            rejected_logps = self.beta * policy_rejected_logps
        else:
            chosen_logps = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_logps = self.beta * (policy_rejected_logps - reference_rejected_logps)
        # calculate the goal
        chosen_logps = torch.sigmoid(chosen_logps)
        rejected_logps = torch.sigmoid(rejected_logps)
        chosen_logps = chosen_logps.to(self.accelerator.device)
        rejected_logps = rejected_logps.to(self.accelerator.device)

        # calculates a conservative DPO loss.
        if self.loss_type == "sigmoid":
            ############################################################################# DUMMY ANSWER
            if chosen_rating >= 0 and rejected_rating >= 0:
                print(1)
                losses = - (torch.log(chosen_logps) + torch.log(1-rejected_logps))
            elif chosen_rating >= 0:
                print(2)
                losses = - (torch.log(chosen_logps))
            elif rejected_rating >= 0:
                print(3)
                losses = - (torch.log(1-rejected_logps))
            #############################################################################
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid']. Currently, other loss types are not supported"
            )

        chosen_rewards = (
            self.beta
            * (
                policy_chosen_logps.to(self.accelerator.device) - reference_chosen_logps.to(self.accelerator.device)
            ).detach()
        )
        rejected_rewards = (
            self.beta
            * (
                policy_rejected_logps.to(self.accelerator.device)
                - reference_rejected_logps.to(self.accelerator.device)
            ).detach()
        )
        return losses, chosen_rewards, rejected_rewards

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_rating,
            rejected_rating,
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_rating,
            rejected_rating,
        )
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return losses.mean(), metrics

# Config
input_data_dir = "./data/"
base_model_name = "meta-llama/Llama-3.1-8B"
padding_side_req = "right" #by default mistral uses left, but trl has overflow issues with left padding.
output_dir = "./una_binary_BCE_0.01"
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
BASE_TEMPLATE = '''<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''
RESPONSE_TEMPLATE = '''{response}<|end_of_text|>'''

def filter_by_length(example, tokenizer):
    len_prompt = len(tokenizer(example["prompt"], padding=False, truncation=False)["input_ids"])
    len_chosen = len(tokenizer(example["chosen"], padding=False, truncation=False)["input_ids"])
    len_rejected = len(tokenizer(example["rejected"], padding=False, truncation=False)["input_ids"])

    return True if (len_prompt <= max_prompt_length and len_prompt + len_chosen <= max_seq_length and len_prompt + len_rejected <= max_seq_length) and (len_prompt != 0) and (len_chosen != 0) and (len_rejected != 0) else False

def apply_formatting(example, tokenizer):
    chosen_response = RESPONSE_TEMPLATE.format(response=example["chosen"][-1]["content"])
    rejected_response = RESPONSE_TEMPLATE.format(response=example["rejected"][-1]["content"])
    prompt = BASE_TEMPLATE.format(prompt=example["prompt"]) 
   
    return {
        "prompt": prompt,
        "chosen": chosen_response,
        "rejected": rejected_response,
        "chosen_rating": 1,
        "rejected_rating": 0
    }

'''
def load_and_format_datasets(input_data_dir, tokenizer):
    dataset_dict_clean = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
    dataset_dict = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    train_dataset = dataset_dict_clean['train'].shuffle(seed=42).map(lambda example: apply_formatting(example, tokenizer))
    dev_dataset = dataset_dict['test_prefs'].map(lambda example: apply_formatting(example, tokenizer))
    
    return train_dataset, dev_dataset
'''

def apply_formatting_helpsteer(example1, example2, tokenizer):
    prompt = BASE_TEMPLATE.format(prompt=example1["prompt"])
    response1 = RESPONSE_TEMPLATE.format(response=example1["response"])
    response2 = RESPONSE_TEMPLATE.format(response=example2["response"])
    score1 = 0.65 * example1["helpfulness"] + 0.8 * example1["correctness"] + 0.45 * example1["coherence"]
    score2 = 0.65 * example2["helpfulness"] + 0.8 * example2["correctness"] + 0.45 * example2["coherence"]
    max_score = 0.65 * 4 + 0.8 * 4 + 0.45 * 4
    score1_norm = score1 / max_score
    score2_norm = score2 / max_score

    if score1_norm > score2_norm:
        return [
            {"prompt": prompt, "chosen": response1, "chosen_rating": score1_norm, "rejected": "DUMMY_ANSWER", "rejected_rating": -1},
            {"prompt": prompt, "chosen": "DUMMY_ANSWER", "chosen_rating": -1, "rejected": response2, "rejected_rating": score2_norm}
        ]
    else:
        return [
            {"prompt": prompt, "chosen": response2, "chosen_rating": score2_norm, "rejected": "DUMMY_ANSWER", "rejected_rating": -1},
            {"prompt": prompt, "chosen": "DUMMY_ANSWER", "chosen_rating": -1, "rejected": response1, "rejected_rating": score1_norm}
        ]

def load_and_format_datasets(input_data_dir, tokenizer):
    dataset_dict_clean = load_dataset("nvidia/HelpSteer2")
    dataset_dict = load_dataset("HuggingFaceH4/ultrafeedback_binarized")

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

    dev_dataset = dataset_dict['test_prefs'].map(lambda example: apply_formatting(example, tokenizer))

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
    tokenizer.padding_side = padding_side_req

    return tokenizer, base_model


def main():
    tokenizer, policy_model = initialize_tokenizer_and_model()
    train_dataset, dev_dataset = load_and_format_datasets(input_data_dir, tokenizer)
    print("Example 0:\nPrompt after preprocessing:", train_dataset[0]["prompt"])
    print("Chosen Response:", train_dataset[0]["chosen"])
    print("Rejected Response:", train_dataset[0]["rejected"])
    print("****"*20)
    print("Example 1:\nPrompt after preprocessing:", dev_dataset[0]["prompt"])
    print("Chosen Response:", train_dataset[1]["chosen"])
    print("Rejected Response:", train_dataset[1]["rejected"])
    # training config
    training_args = TrainingArguments(
        output_dir=output_dir,
        warmup_ratio=0.1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
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
        run_name=f"logs-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )
    # ref model would be a clone of policy model if not specificed
    # trl takes care of its initialization
    dpo_trainer = UNA_BCE(
        model=policy_model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        peft_config=peft_config,
        max_length=max_seq_length,
        max_prompt_length=max_prompt_length,
        tokenizer=tokenizer,
        args=training_args, 
        beta=0.01
    )

    dpo_trainer.train()
    #dpo_trainer.save_model(os.path.join(output_dir, "save_model"))
    # Save artifacts
    dpo_trainer.model.save_pretrained(os.path.join(output_dir, "final_checkpoint"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_checkpoint"))
    # Flush memory
    del dpo_trainer, policy_model
    gc.collect()
    torch.cuda.empty_cache()
    
    
if __name__ == "__main__":
    main()