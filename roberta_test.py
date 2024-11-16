import pandas as pd

# Load the uploaded dataset
file_path = '/Users/apple/Desktop/CSMU_LAB/playground/IMDB Dataset.csv'
imdb_data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
print(imdb_data.head())


from sklearn.model_selection import train_test_split
from datasets import Dataset

# Preprocessing: Map 'sentiment' to binary labels
imdb_data['label'] = imdb_data['sentiment'].map({'positive': 1, 'negative': 0})

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    imdb_data['review'], imdb_data['label'], test_size=0.2, random_state=42
)

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
test_dataset = Dataset.from_pandas(pd.DataFrame({'text': test_texts, 'label': test_labels}))

# Display the number of samples in train and test datasets
print(len(train_dataset), len(test_dataset))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import torch

# 加載 RoBERTa 分詞器和模型
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# 預處理數據
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# 定義訓練參數
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1
)

# 計算指標
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, logits[:, 1])
    return {"accuracy": acc, "roc_auc": auc}

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# 開始訓練
trainer.train()

# 儲存模型
model.save_pretrained("./roberta_sentiment_model")
tokenizer.save_pretrained("./roberta_sentiment_model")

# 測試模型並獲取預測
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
labels = predictions.label_ids
probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()

# 繪製 ROC 曲線
fpr, tpr, thresholds = roc_curve(labels, probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(labels, probs):.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# 加載微調後的模型和分詞器
tokenizer = AutoTokenizer.from_pretrained("./roberta_sentiment_model")
model = AutoModelForSequenceClassification.from_pretrained("./roberta_sentiment_model")

# 輸入新的影評進行預測
new_reviews = [
    "The movie was fantastic! I loved the characters and the story.",
    "It was a terrible movie. I regret watching it."
]

# 分詞處理並進行預測
inputs = tokenizer(new_reviews, return_tensors="pt", truncation=True, padding=True, max_length=128)
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)

# 輸出結果
for review, prob in zip(new_reviews, probs):
    sentiment = "positive" if torch.argmax(prob) == 1 else "negative"
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}, Confidence: {prob.max().item():.2f}")
