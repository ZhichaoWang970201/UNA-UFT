# Model Training & Evaluation

This repository contains scripts and configurations for training and evaluating a language model using [Hugging Face Accelerate](https://github.com/huggingface/accelerate) and [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness).

---

## 🔐 Authentication

Before pulling or pushing models to Hugging Face Hub, log in:

```bash
huggingface-cli login --token YOUR_TOKEN_HERE
```

## 🏋️‍♂️ Training the Model
We use accelerate to launch multi-GPU training sessions. There are five files to be trained "dpo_trainer_0.03.py", "kto_trainer_0.03.py", "UNA_trainer_binary_BCE_0.01.py", "UNA_trainer_binary_MSE_0.01.py" and "UNA_trainer_score_MSE_3e-5_0.03.py".

```bash
export ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file multi_gpu.yaml \
  UNA_trainer_binary_MSE_0.01.py
```


📝 Note on TRL Versions

Use trl==0.10.1 when running kto_trainer_0.03.py

Use trl==0.8.6 for all other training scripts

## 📊 Evaluation
We use the lm_eval harness to benchmark across various tasks with customizable few-shot settings.

### Few-Shot Mapping

| Task                 | # Few-Shot Examples |
|----------------------|---------------------|
| leaderboard_bbh      | 3                   |
| leaderboard_gpqa     | 0                   |
| leaderboard_mmlu_pro | 5                   |
| leaderboard_musr     | 0                   |
| leaderboard_ifeval   | 0                   |
| leaderboard_math_hard| 4                   |
| gsm8k                | 5                   |
| truthfulqa           | 0                   |
| winogrande           | 5                   |
| arc_challenge        | 25                  |
| hellaswag            | 10                  |
| mmlu                 | 5                   |

Import this mapping into your evaluation script:

```bash
fewshot_mapping = {
    'leaderboard_bbh': 3,
    'leaderboard_gpqa': 0,
    'leaderboard_mmlu_pro': 5,
    'leaderboard_musr': 0,
    'leaderboard_ifeval': 0,
    'leaderboard_math_hard': 4,
    'gsm8k': 5,
    'truthfulqa': 0,
    'winogrande': 5,
    'arc_challenge': 25,
    'hellaswag': 10,
    'mmlu': 5,
}
```

✅ Running Evaluation
Replace \<model_name\>, \<adapter\>, \<task\>, and \<num_fewshot\> with your desired values:

```bash
accelerate launch -m lm_eval \
  --model hf \
  --model_args pretrained=<model_name>,peft={adapter},trust_remote_code=True,dtype=bfloat16 \
  --tasks <task> \
  --num_fewshot <num_fewshot> \
  --batch_size auto:1
```
