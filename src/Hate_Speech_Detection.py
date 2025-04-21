# Install required packages (for standalone execution)
# Uncomment if running in a fresh environment
!pip install datasets -q
!pip install nlpaug -q
!pip install nltk -q
!pip install wandb -q

# Import necessary libraries
import pandas as pd  # Ref [1]: McKinney, W., "pandas: a foundational Python library for data analysis and statistics," Python for Data Analysis, 2010.
import numpy as np  # Ref [2]: Harris, C. R., et al., "Array programming with NumPy," Nature, vol. 585, pp. 357–362, 2020.
import re
import nltk  # Ref [3]: Bird, S., et al., "Natural Language Processing with Python," O'Reilly Media, 2009.
from sklearn.model_selection import train_test_split  # Ref [4]: Pedregosa, F., et al., "Scikit-learn: Machine Learning in Python," JMLR, vol. 12, pp. 2825–2830, 2011.
from sklearn.feature_extraction.text import TfidfVectorizer  # Ref [4]
from sklearn.linear_model import LogisticRegression  # Ref [4]
from sklearn.svm import SVC  # Ref [4]
from sklearn.ensemble import RandomForestClassifier  # Ref [4]
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, f1_score  # Ref [4]
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments  # Ref [5]: Wolf, T., et al., "Transformers: State-of-the-Art Natural Language Processing," in Proc. EMNLP, 2020, pp. 38–45.
from datasets import Dataset  # Ref [6]: Lhoest, Q., et al., "Datasets: A Community Library for Natural Language Processing," 2021. [Online]. Available: https://github.com/huggingface/datasets
import torch  # Ref [7]: Paszke, A., et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library," in NeurIPS, 2019, pp. 8024–8035.
import wandb  # Ref [8]: Biewald, L., "Weights & Biases: Experiment Tracking for Machine Learning," Weights & Biases, 2020. [Online]. Available: https://wandb.ai
from nlpaug.augmenter.word import SynonymAug  # Ref [9]: Ma, E., "NLP Augmentation," 2019. [Online]. Available: https://github.com/makcedward/nlpaug
from sklearn.utils.class_weight import compute_class_weight  # Ref [4]

# Download NLTK resources
nltk.download('wordnet')

# Initialize W&B
wandb.init(project="hate_speech_detection", name="roberta_optimized")

# Verify GPU
print("CUDA Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True

# Mount Google Drive to access dataset files
from google.colab import drive
drive.mount('/content/drive')

# Load Dataset
train_dataset_path = "/content/drive/MyDrive/Hate Speech Detection/train_en.tsv"
test_dataset_path = "/content/drive/MyDrive/Hate Speech Detection/test_en.tsv"
df = pd.read_csv(train_dataset_path, sep="\t")  # Ref [10]: Basile, V., et al., "SemEval-2019 Task 5," in Proc. SemEval, 2019, pp. 54–63.
df_test = pd.read_csv(test_dataset_path, sep="\t", header=None)
df_test.columns = ['id', 'text', 'HS', 'TR', 'AG']

# Enhanced Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^\w\s#@]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)

# Light Data Augmentation
aug = SynonymAug(aug_p=0.2)  # 20% synonym replacement - Ref [9]
augmented_texts = []
augmented_labels = []

for idx, row in df.iterrows():
    augmented_texts.append(row['clean_text'])
    augmented_labels.append(row['HS'])
    if row['HS'] == 1 and np.random.random() < 0.5:  # Augment 50% of Hate Speech samples
        try:
            augmented_text = aug.augment(row['clean_text'])[0]
            augmented_texts.append(augmented_text)
            augmented_labels.append(row['HS'])
        except Exception as e:
            print(f"Augmentation error for row {idx}: {e}")
            augmented_texts.append(row['clean_text'])
            augmented_labels.append(row['HS'])

df_augmented = pd.DataFrame({'clean_text': augmented_texts, 'HS': augmented_labels})

# Verify augmentation
print("Training set size after augmentation:", len(df_augmented))
print("Label Distribution After Augmentation:\n", df_augmented['HS'].value_counts())

# Vectorization for Traditional ML
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)  # Ref [4]
X = vectorizer.fit_transform(df_augmented['clean_text'])
y = df_augmented['HS']
X_test = vectorizer.transform(df_test['clean_text'])

# Train-Test Split for Traditional ML
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Traditional Model Training & Evaluation
models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=500),
    "SVM": SVC(C=1.0, kernel='linear', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{name} Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
    print(classification_report(y_val, y_pred))
    test_pred = model.predict(X_test)
    df_test[f'{name}_Prediction'] = test_pred

df_test.to_csv("test_predictions.csv", index=False)

# Deep Learning Setup with RoBERTa
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # Ref [5]
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Precompute Tokenization
train_data = Dataset.from_pandas(df_augmented[['clean_text', 'HS']])
test_data = Dataset.from_pandas(df_test[['clean_text', 'HS']])

def tokenize_function(examples):
    return tokenizer(examples['clean_text'], padding='max_length', truncation=True, max_length=64)

tokenized_train = train_data.map(tokenize_function, batched=True, batch_size=32)
tokenized_test = test_data.map(tokenize_function, batched=True, batch_size=32)

# Split tokenized_train
train_val_split = tokenized_train.train_test_split(test_size=0.2, seed=42)
tokenized_train = train_val_split['train']
tokenized_val = train_val_split['test']

# Debug: Verify tokenized_val
print("Tokenized Val Dataset Columns Before Formatting:", tokenized_val.column_names)
print("Tokenized Val Dataset Sample Before Formatting:", tokenized_val[0] if tokenized_val else "Empty dataset")
print("Tokenized Val Dataset Length:", len(tokenized_val))

# Rename and format
tokenized_train = tokenized_train.rename_column("HS", "labels")
tokenized_val = tokenized_val.rename_column("HS", "labels")
tokenized_test = tokenized_test.rename_column("HS", "labels")
tokenized_train = tokenized_train.remove_columns(['clean_text'])
tokenized_val = tokenized_val.remove_columns(['clean_text'])
tokenized_test = tokenized_test.remove_columns(['clean_text'])
tokenized_train.set_format('torch')
tokenized_val.set_format('torch')
tokenized_test.set_format('torch')

# Compute class weights
train_labels = df_augmented['HS'].values
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to('cuda' if torch.cuda.is_available() else 'cpu')

# Define compute_metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": accuracy, "f1": f1}

# TrainingArguments
training_args = TrainingArguments(
    output_dir='/content/drive/MyDrive/Hate Speech Detection/bert_hate_speech_model',
    eval_strategy='epoch',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir='./logs_roberta',
    logging_steps=10,
    save_strategy='epoch',
    load_best_model_at_end=True,
    gradient_accumulation_steps=1,
    fp16=True,
    report_to="wandb",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

# Train the model
print("Starting training...")
trainer.train()
print("Training completed.")
eval_results = trainer.evaluate()
print("RoBERTa Evaluation Results:", eval_results)
wandb.log({"validation_loss": eval_results['eval_loss'], "validation_accuracy": eval_results['eval_accuracy']})

# Threshold Optimization
print("Optimizing threshold...")
predictions = trainer.predict(tokenized_val)
probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1)[:, 1].numpy()
true_labels = tokenized_val['labels'].numpy()
precision, recall, thresholds = precision_recall_curve(true_labels, probs)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal Threshold: {optimal_threshold:.4f}")
print(f"Max F1-Score: {f1_scores[optimal_idx]:.4f}")

# Test Predictions
print("Making test predictions...")
test_predictions = trainer.predict(tokenized_test)
probs_test = torch.softmax(torch.tensor(test_predictions.predictions), dim=-1)[:, 1].numpy()
pred_labels_test = (probs_test >= optimal_threshold).astype(int)
accuracy_test = accuracy_score(tokenized_test['labels'].numpy(), pred_labels_test)
print(f"RoBERTa Optimized Accuracy on Test Set: {accuracy_test * 100:.2f}%")
print(classification_report(tokenized_test['labels'].numpy(), pred_labels_test, target_names=['Non-Hate Speech', 'Hate Speech']))
wandb.log({"test_accuracy": accuracy_test})

# Backup the Model
!zip -r "/content/best_model.zip" "/content/drive/MyDrive/Hate Speech Detection/bert_hate_speech_model"
from google.colab import files
files.download("/content/best_model.zip")

# Finish W&B run
wandb.finish()
print("Script execution completed.")
