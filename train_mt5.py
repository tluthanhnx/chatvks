from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "google/mt5-small"  # Nhẹ hơn rất nhiều
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset từ JSON
data = load_dataset(
    'json',
    data_files={
        'train': 'train.json',
        'validation': 'dev.json'  # cần có ít nhất 1 mẫu trong dev.json
    }
)

# Tokenize dữ liệu
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
    per_device_train_batch_size=2,     # 👈 Giảm batch size
    num_train_epochs=10,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    weight_decay=0.01,
    fp16=False,                        # 👈 Tắt FP16 nếu dùng CPU
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_enc['train'],
    eval_dataset=data_enc['validation'],
)

trainer.train()
from transformers import MT5Tokenizer
trainer.save_model("model/vietext2sql_mt5")
