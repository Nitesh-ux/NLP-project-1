from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# -----------------------------
# 1. Load Model + Tokenizer
# -----------------------------
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# -----------------------------
# 2. Dummy Dataset (Replace with your own)
# -----------------------------
train_texts = [
    {"text": "Hello, I am fine-tuning."},
    {"text": "This is another example."}
]

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True)

dataset = Dataset.from_list(train_texts)
tokenized_ds = dataset.map(tokenize)

# -----------------------------
# 3. Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./finetuned_llama",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=50,
)

# -----------------------------
# 4. Train Model
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds,
)

trainer.train()

# -----------------------------
# 5. Save Final Model
# -----------------------------
trainer.save_model("./finetuned_llama")
print("Training Completed & Model Saved!")
