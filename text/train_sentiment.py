import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    BertTokenizer,
    ErnieForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from torch.utils.data import Dataset

# ================= 配置区域 =================
DATA_PATH = "E:/AI_Customer/models/datasets/balanced_data.csv"
MODEL_NAME = "E:/AI_Customer/models/nghuyong--ernie-3.0-base-zh"
label_map = {0: "负面", 1: "中性", 2: "正面"}
num_labels = len(label_map)
BATCH_SIZE = 16
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
OUTPUT_DIR = "./sentiment_ernie_v1"

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, truncation=True, max_length=self.max_length,
            return_tensors='pt', padding=False
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    return {"accuracy": acc, "f1_macro": f1}

def main():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("Using CPU (slow)")
        device = torch.device("cpu")

    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = ErnieForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels).to(device)

    train_dataset = SentimentDataset(train_df, tokenizer, MAX_LENGTH)
    test_dataset = SentimentDataset(test_df, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    print("Training...")
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done.")

if __name__ == "__main__":
    main()
