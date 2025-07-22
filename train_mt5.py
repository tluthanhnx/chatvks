from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "google/mt5-small"  # Nhẹ hơn để huấn luyện
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset từ JSON
data = load_dataset(
    'json',
    data_files={
        'train': 'train.json',
        'validation': 'dev.json'  # cần ít nhất 1 mẫu
    }
)

# Tiền xử lý
def preprocess(batch):
    inputs = tokenizer(batch['question'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(batch['sql'], max_length=256, truncation=True, padding="max_length")
    inputs['labels'] = labels['input_ids']
    return inputs

data_enc = data.map(preprocess, batched=True, remove_columns=['question', 'sql'])

# Huấn luyện
training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,     # 👈 Dễ huấn luyện
    num_train_epochs=10,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    weight_decay=0.01,
    fp16=False,                        # 👈 Tắt nếu không có GPU
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_enc['train'],
    eval_dataset=data_enc['validation'],
)

# Train mô hình
trainer.train()

# ✅ Lưu mô hình và tokenizer
output_dir = "./vncorenlp/models/vietext2sql_mt5"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
