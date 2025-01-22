
# Unified Alignment and Fine-Tuning Framework: UNA & UFT

Welcome to the repository for **UNA** and **UFT**, two groundbreaking frameworks for advancing large language model (LLM) alignment and fine-tuning. These methodologies aim to simplify, stabilize, and enhance the training process for LLMs, addressing key challenges in RLHF, DPO, KTO, and supervised fine-tuning.

## 🚀 Highlights

- **UNA (Unified Alignment):**
  - Unifies RLHF/PPO, DPO, and KTO into a single supervised learning framework.
  - Simplifies and stabilizes alignment training while reducing computational requirements.
  - Handles diverse feedback types, including pairwise, binary, and scalar feedback.
  - Demonstrated superior performance on downstream tasks compared to existing approaches.

- **UFT (Unified Fine-Tuning):**
  - Combines supervised fine-tuning (SFT) and alignment into a unified training stage.
  - Addresses catastrophic forgetting by integrating instruction-tuning and alignment data.
  - Establishes an efficient pretraining-UFT paradigm for LLMs.
  - Outperforms traditional SFT and sequential training pipelines in both alignment and generation tasks.

---

## 📘 Papers

1. **[UNA: Unifying Alignments of RLHF/PPO, DPO, and KTO by a Generalized Implicit Reward Function](./UNA.pdf)**  
   RLHF and DPO limitations are tackled by a generalized implicit reward function. UNA simplifies reinforcement learning, accelerates training, and reduces memory overhead while improving performance.

2. **[UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA](./UFT.pdf)**  
   This framework eliminates the gap between pretraining and alignment by unifying SFT and alignment stages, improving downstream performance and avoiding catastrophic forgetting.

---

## 🛠️ Installation and Setup

To replicate the experiments, the authors used **AWS SageMaker** with the `pytorch_p310` base image. Follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/ZhichaoWang970201/UNA-UFT.git
   cd UFT-UNA
   ```

2. Directly run the notebook
   
---

## 🌟 Key Contributions

1. **UNA**: Simplifies RLHF by replacing reinforcement learning with stable supervised learning.
2. **UFT**: Unifies SFT and alignment into one stage, enhancing efficiency and preventing forgetting.
3. **Flexibility**: Handles multiple feedback types and operates in both online and offline modes.

---

## 📝 Citation

If you use this repository or its concepts in your research, please cite the respective papers:

```bibtex
@misc{wang2024unaunifyingalignmentsrlhfppo,
      title={UNA: Unifying Alignments of RLHF/PPO, DPO and KTO by a Generalized Implicit Reward Function}, 
      author={Zhichao Wang and Bin Bi and Can Huang and Shiva Kumar Pentyala and Zixu James Zhu and Sitaram Asur and Na Claire Cheng},
      year={2024},
      eprint={2408.15339},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.15339}, 
}

@misc{wang2024uftunifyingfinetuningsft,
      title={UFT: Unifying Fine-Tuning of SFT and RLHF/DPO/UNA through a Generalized Implicit Reward Function}, 
      author={Zhichao Wang and Bin Bi and Zixu Zhu and Xiangbo Mao and Jun Wang and Shiyu Wang},
      year={2024},
      eprint={2410.21438},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.21438}, 
}
```
