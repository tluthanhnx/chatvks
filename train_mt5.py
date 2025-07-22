from transformers import MT5ForConditionalGeneration, MT5Tokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Load pre-trained model and tokenizer
model_name = "google/mt5-base"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# Load dataset from JSON files
data = load_dataset('json', data_files={'train': 'train.json', 'validation': 'dev.json'})

# Preprocessing function
def preprocess(batch):
    inputs = tokenizer(batch['question'], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(batch['sql'], max_length=256, truncation=True, padding="max_length")
    inputs['labels'] = labels['input_ids']
    return inputs

# Apply preprocessing
data_enc = data.map(preprocess, batched=True, remove_columns=['question', 'sql'])

# Define training arguments
training_args = TrainingArguments(
    output_dir="output",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    num_train_epochs=50,
    save_strategy="epoch",
    logging_steps=50,
    save_total_limit=2,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data_enc['train'],
    eval_dataset=data_enc['validation'],
)

# Train and save model
trainer.train()
trainer.save_model("output/best_model")