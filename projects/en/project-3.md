# NLP/LLM Project: Dialogue Summarization

**Developed a model that quickly summarizes various Korean conversations from daily life. Achieved 1st place on the leaderboard by comparing Encoder-Decoder architecture-based models with Decoder-only architecture-based LLM models through data augmentation and QLoRA fine-tuning.**

## Project Overview

This project aims to develop a Korean dialogue summarization model that accurately and concisely summarizes the essence of daily conversations.

GitHub Repository: [https://github.com/jkim1209/NLI-Dialogue-Summarization](https://github.com/jkim1209/NLI-Dialogue-Summarization)

Presentation Slides: [Google Slides](https://docs.google.com/presentation/d/1n3ZpdBC2U84vmS9k6mGyuqrokve7uZQvnuWrouwvFv4)

## My Role and Contributions

As team leader, I oversaw the entire project and led the LLM (Large Language Model) modeling aspect.

- Data Strategy Development: Designed and applied data augmentation techniques such as Paraphrase and Speaker Swap to improve the model's generalization performance.
- LLM Modeling & Fine-tuning: Selected Decoder-only architectures (SOLAR, Qwen3, etc.) and performed efficient fine-tuning in resource-constrained environments using QLoRA techniques.
- Prompt Engineering: Designed and tested various prompt structures to enable the model to generate optimal summaries.
- Performance Analysis & Improvement: Compared and analyzed ROUGE scores and qualitative differences in generated outputs between T5 models and LLMs to select the final model.

## Key Technologies and Implementation

### Technologies Used

- Framework: Python, PyTorch, Transformers, PEFT, TRL, BitsAndBytes
- Models: T5, KoBART, SOLAR 10.7B, Qwen3 0.6B

### Core Implementation

- Built data augmentation pipeline (Paraphrase, Speaker Swap, Synthetic Generation)
- Comparative experiments between Encoder-Decoder (T5, KoBART) and Decoder-only (LLM) models
- 4-bit quantization and fine-tuning using QLoRA technique

![Data Augmentation Pipeline](/projects/assets/images/03/01.png)

## Troubleshooting

### Problem: VRAM Shortage During LLM Fine-tuning

**Problem Description**

Unlike Encoder-Decoder models, LLM models require significantly more tokens for Korean text processing, and when combined with instruction prompts for summarization, token count surges. This caused repeated OOM errors and kernel crashes due to VRAM(24GB) shortage during QLoRA fine-tuning.

**Solution**

1. Minimized Batch Size: Set Batch Size to 1 to minimize memory footprint of single training samples. Applied Gradient Accumulation technique to achieve the same effect as the previous batch size.
2. Optimized Input Length: Reduced max_token length to 4096 to cover most conversation lengths, and applied zero-shot for excessively long conversations during training.
3. As a result, successfully performed fine-tuning even in resource-constrained environments.

## Results and Achievements

### Quantitative Results

Achieved 1st place in the daily conversation summarization competition

- Private ROUGE: 47.9550
- Baseline ROUGE: 15.8301

### Learnings and Insights

- Gained experience in successfully fine-tuning large language models in resource-constrained environments like consumer GPUs using QLoRA.
- Learned the importance of balancing quantitative metrics like ROUGE scores with qualitative aspects like the naturalness of summaries as perceived by humans during LLM fine-tuning.
- Developed collaborative decision-making skills by discussing various approaches with team members throughout the modeling process and making optimal decisions based on objective evidence.

![Competition Result](/projects/assets/images/03/02.png)
