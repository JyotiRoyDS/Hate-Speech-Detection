# Hate-Speech-Detection
A machine learning project to detect hate speech using NLP 
--------
This repository contains the code for hate speech detection using the SemEval 2019 Task 5 dataset [1]. The project implements two main tasks:

1. Task 1: Descriptive Analysis of the Dataset - Computes statistics (dataset size, class distribution, word count) and generates visualizations (class distribution bar chart, word cloud) to complement the "Dataset Analysis" section of our report.

2. Task 2: Preprocessing, Training, and Evaluation - Preprocesses the dataset, trains traditional machine learning models (Logistic Regression, SVM, Random Forest) and a RoBERTa model on the training set, and evaluates them on the test set, as reported in the "Methodology", "Experimental Setting", and "Results" sections of our report.

After threshold optimization, the RoBERTa model achieved a validation accuracy of 84.93% (surpassing the 75% benchmark) and a test macro F1-score of 0.8635. Runtime is approximately 5:49 minutes on a T4 GPU. Optionally, Weights & Biases (W&B) can be enabled to log training metrics to a dashboard.

Files in This Repository
------------------------
- "generate_visualizations.py": Script for Task 1 - Computes dataset statistics and generates visualizations.

-"hate_speech_detection.py" is the script for Task 2, which preprocesses the dataset, trains models, and evaluates them.

- bert_hate_speech_model/: Directory with RoBERTa model configuration files (weights excluded due to size):
  - config.json: Model configuration
  - special_tokens_map.json: Special tokens mapping
  - tokenizer_config.json: Tokenizer configuration
  - vocab.txt: Tokenizer vocabulary

- requirements.txt: List of Python dependencies

How to Run the Code
-------------------

Prerequisites
-------------
1. Python Version: 3.8 or higher
2. Hardware: Developed on Google Colab with a T4 GPU (recommended for 5:49-minute runtime)
3. Dataset: SemEval 2019 Task 5 dataset (train_en.tsv, test_en.tsv) required - see "Dataset Access" below

Dependencies
------------
Install dependencies from requirements.txt:
  pip install -r requirements.txt

Contents of requirements.txt:
  pandas==1.5.3
  numpy==1.24.3
  scikit-learn==1.3.0
  nltk==3.8.1
  nlpaug==1.1.11
  transformers==4.35.0
  torch==2.0.1
  datasets==2.14.0
  matplotlib==3.7.1
  wordcloud==1.9.3
  seaborn==0.12.2
  wandb==0.15.8

Weights & Biases (W&B) Setup 
-----------------------------
W&B logs training metrics (e.g., validation loss, accuracy) to a dashboard.

To enable W&B:

1. Sign up at https://wandb.ai/ and get your API key from https://wandb.ai/settings

2. In Google Colab:
   - Run hate_speech_detection.py; enter your API key when prompted
   - Or set it programmatically before running:
     import os
     os.environ["WANDB_API_KEY"] = "your-api-key-here"

3. Modify hate_speech_detection.py (line 159) to enable W&B:
     training_args = TrainingArguments(
         output_dir='/content/drive/MyDrive/Hate Speech Detection/bert_hate_speech_model',
         report_to="wandb",  # Enable W&B logging
         ...
     )

4. Add W&B code (after imports, around line 25):
     import wandb
     wandb.init(project="hate_speech_detection", name="roberta_optimized")

5. Add logging after evaluation (around line 175):
     wandb.log({"validation_loss": eval_results['eval_loss'], "validation_accuracy": eval_results['eval_accuracy']})

6. Add logging after test predictions (around line 195):
     wandb.log({"test_accuracy": accuracy_test, "test_f1": test_f1})

7. Add at the end (around line 205):
     wandb.finish()

To skip W&B, keep report_to="none" (default) and omit steps 4-7.

Dataset Access
--------------
- Source: SemEval 2019 Task 5 [1]
- Files: train_en.tsv (10,880 samples after augmentation), test_en.tsv (2,971 samples)
- Download: https://competitions.codalab.org/competitions/19935
- Placement: Store in /content/drive/MyDrive/Hate Speech Detection/ (Colab) or update paths in scripts:
  - generate_visualizations.py (lines 22-23)
  - hate_speech_detection.py (lines 51-52)

Environment Setup
-----------------
1. Google Colab (Recommended):

   - Open https://colab.research.google.com/

   - Mount Google Drive:
     from google.colab import drive
     drive.mount('/content/drive')

   - Create Hate Speech Detection folder:
     - In Google Drive (https://drive.google.com/), create /My Drive/Hate Speech Detection/

     - Or in Colab:
       import os
       os.makedirs('/content/drive/MyDrive/Hate Speech Detection/bert_hate_speech_model', exist_ok=True)

   - Update output_dir (if needed) in hate_speech_detection.py (line 168):
       training_args = TrainingArguments(
           output_dir='/your/custom/path/bert_hate_speech_model',
           ...
       )

   - Upload scripts, requirements.txt, and dataset to Colab

   - Enable GPU: Runtime > Change runtime type > T4 GPU

   - Install dependencies:
     !pip install -r requirements.txt

2. Local Machine:

   - Install Python 3.8+

   - Install dependencies: pip install -r requirements.txt

   - GPU recommended (CPU slower)

   - Update dataset paths in scripts (lines as above)

   - Update output_dir in hate_speech_detection.py (line 168):
       training_args = TrainingArguments(
           output_dir='./results',
           ...
       )


Steps to Run
------------

Task 1: Descriptive Analysis of the Dataset
------------------------------------------

Script: generate_visualizations.py

Run: python generate_visualizations.py

What It Does:
- Loads train_en.tsv and test_en.tsv

- Computes:
  - Dataset size (train: 7200, test: 2971 before augmentation)
  - Class distribution (HS=0 vs HS=1)
  - Missing values
  - Average word count

- Generates:
  - class_distribution.png (bar chart, Figure 1)
  - word_cloud.png (word cloud of HS=1 terms, Figure 2)

Expected Output:
----------------
  Train Dataset Shape: (7200, [columns])
  Test Dataset Shape: (2971, 5)
  Sample of Train Data:
  [5 rows]
  Sample of Test Data:
  [5 rows]
  Training set size: 7200
  Test set size: 2971
  Training set class distribution: HS=0: 5217, HS=1: 1983
  Test set class distribution: HS=0: 1719, HS=1: 1252
  Missing Values in Training Set:
  [counts]
  Missing Values in Test Set:
  [counts]
  Average word count (train): [value]
  Average word count (test): [value]
  Descriptive analysis completed. Visualizations saved as 'class_distribution.png' and 'word_cloud.png'.

Files:
- class_distribution.png
- word_cloud.png

Task 2: Preprocessing, Training, and Evaluation
-----------------------------------------------

Script: hate_speech_detection.py

Run: python hate_speech_detection.py

What It Does:
- Preprocessing:
  - Loads train_en.tsv and test_en.tsv
  - Cleans text (lowercase, remove URLs, special chars, spaces)
  - Augments 50% of HS=1 samples (train size: 10,880)

- Traditional ML:
  - Trains Logistic Regression, SVM, Random Forest with TF-IDF
  - Evaluates on validation and test sets

- RoBERTa:
  - Splits train data (80/20)
  - Trains roberta-base (3 epochs, batch size 8, lr 2e-5, fp16)
  - Evaluates on validation and test sets
  - Optimizes threshold for F1-score

- W&B Logging:
  - Logs metrics to hate_speech_detection project

Expected Output:
------------------
  Training set size after augmentation: 10880
  Label Distribution After Augmentation:
  0    5217
  1    5663
  Logistic Regression Validation Accuracy: [value]
  [report]
  SVM Validation Accuracy: [value]
  [report]
  Random Forest Validation Accuracy: [value]
  [report]
  Starting training...
  [logs for 3 epochs]
  Training completed.
  RoBERTa Evaluation Results: {'eval_loss': [value], 'eval_accuracy': 0.8493, 'eval_f1': [value], ...}
  Optimizing threshold...
  Optimal Threshold: 0.3413
  Max F1-Score: [value]
  Making test predictions...
  RoBERTa Optimized Accuracy on Test Set: 46.79%
  RoBERTa Optimized Test Macro F1-Score: 0.8635
  [report]

W&B Output:
  - Metrics logged to https://wandb.ai/your-username/hate_speech_detection:
    validation_loss: [value]
    validation_accuracy: 0.8493
    test_accuracy: 0.4679
    test_f1: 0.8635

Files:
- test_predictions.csv (ML predictions)
- bert_hate_speech_model/ (model weights)
- best_model.zip (zipped model)
Runtime: ~5:49 minutes (T4 GPU)

Notes
-----
- GPU Memory: Reduce batch size (lines 161-162) if needed
- Dataset Issues: See Appendix A of proforma
- Model Weights: Regenerate or download from [link]
- W&B: Optional; disable with report_to="none"

References
----------
[1] V. Basile et al., "SemEval-2019 Task 5," in Proc. SemEval-2019, pp. 54–63, doi: 10.18653/v1/S19-2007.
See G18_Report.pdf for full list.

