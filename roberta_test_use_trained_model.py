from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加載保存的模型和分詞器
model_dir = "./roberta_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
# 輸入新的影評
new_reviews = [
    "This movie was absolutely fantastic! I loved it.",
    "Terrible film, I wasted my time watching this."
]

# 將輸入轉換為模型的格式
inputs = tokenizer(new_reviews, return_tensors="pt", padding=True, truncation=True, max_length=128)

# 獲取模型的輸出
outputs = model(**inputs)
probs = torch.softmax(outputs.logits, dim=1)

# 解析結果
for review, prob in zip(new_reviews, probs):
    sentiment = "positive" if torch.argmax(prob) == 1 else "negative"
    confidence = prob.max().item()
    print(f"Review: {review}")
    print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")