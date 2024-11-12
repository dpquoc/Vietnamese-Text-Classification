# Vietnamese Text Classification

## Overview
This project implements a text classification model using a BERT-based architecture to classify Vietnamese text. The model used is **PhoBERT**, which is built on the **RoBERTa** architecture and pre-trained on large Vietnamese corpora. It is fine-tuned in this project to classify offensive and hate speech in Vietnamese.

The dataset used for this project is **ViHOS** (Vietnamese Hate and Offensive Spans Detection). Although ViHOS is designed for span detection (identifying spans of offensive or hateful content), it has been adapted here for binary classification. In this project, we classify the entire text as either **offensive** or **non-offensive**.

## Dataset
- **ViHOS Dataset**: Originally created for span detection, ViHOS pinpoints spans of offensive content in Vietnamese text. For this project, the dataset has been modified into a binary classification format where each text entry is labeled as either offensive or non-offensive. The dataset provides real-world examples for toxic content classification.

## Training and Evaluation
The PhoBERT model was fine-tuned using the modified ViHOS dataset. Training was conducted to adjust the model’s pre-trained weights for binary classification, determining whether a given text is offensive or non-offensive.

Once trained, the model was evaluated on a test dataset, yielding strong results in both accuracy and F1 score.

## Results
- **Accuracy**: 90.14%
- **F1 Score**: 89.49%