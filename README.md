# Emotion Detection from Text Using BERT

This project implements an emotion classification model using the `bert-base-uncased` transformer to detect 13 distinct emotions from tweets. It addresses challenges like class imbalance and noisy data through various augmentation strategies.

---

##Project Overview

Emotion recognition from text is a challenging NLP task, especially on social media data where language is informal and emotions are subtle. This project explores and compares the following techniques for improving emotion classification accuracy:

- **Synonym-based Data Augmentation**
- **AI-Synthesized Data Augmentation using ChatGPT**
- **Filtering to Only Prominent Classes**

The best-performing model was achieved through AI-synthesized tweet generation, leading to improved performance across multiple metrics.

---

## Dataset

- **Name**: [Emotion Detection from Text](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
- **Size**: ~40,000 tweets
- **Labels**: 13 emotion classes  
  `['anger', 'boredom', 'empty', 'enthusiasm', 'fun', 'happiness', 'hate', 'love', 'neutral', 'relief', 'sadness', 'surprise', 'worry']`

---

## Techniques Used

| Stage | Method |
|-------|--------|
| Data Cleaning | Special character removal, stopword filtering, casual term normalization |
| Augmentation | WordNet synonyms, ChatGPT-generated samples |
| Tokenization | HuggingFace BERT tokenizer |
| Modeling | Fine-tuned `bert-base-uncased` with custom classification head |
| Optimization | Layer-wise LR decay, class-weighted loss, AdamW optimizer, gradient accumulation |

---

## Results Summary

| Approach                         | Accuracy | F1 Score |
|----------------------------------|----------|----------|
| Synonym-Based Augmentation       | 44%      | 47%      |
| **AI-Synthesized Data (Best)**   | **48%**  | **48%**  |
| Prominent Class Filtering        | 46%      | 46%      |

---

## Model Architecture

- BERT embedding layer
- Dense layer (ReLU, 128 units)
- Dropout (0.5)
- Softmax output (13-class classification)

Fine-tuning was done in **two phases**:
1. Train classification head only
2. Unfreeze last 4 BERT layers and continue fine-tuning

---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
