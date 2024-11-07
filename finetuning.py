import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
from transformers import DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import bitsandbytes as bnb

# Specify the cache directory
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/falcon'

# Load the tokenizer and model with the specified cache directory
model_name = 'tiiuae/falcon-40b-instruct'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,           # Set cache directory for tokenizer
    trust_remote_code=True
)

# Load the model with 8-bit precision to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,           # Set cache directory for model
    device_map='auto',             # Automatically map model to available devices
    torch_dtype=torch.float16,     # Use float16 for faster training
    load_in_8bit=True,             # Load model in 8-bit precision
    trust_remote_code=True
)

# Apply LoRA for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                # Rank of the update matrices
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05    # Dropout for LoRA layers
)

model = get_peft_model(model, peft_config)

# Function to load data from JSONL file
def load_jsonl(file_path):
    prompts = []
    completions = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompts.append(entry['prompt'])
            completions.append(entry['completion'])
    return {'prompt': prompts, 'completion': completions}

# Load your training data
data = load_jsonl('fine_tuning_training_data.jsonl')
dataset = Dataset.from_dict(data)

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['prompt']
    outputs = examples['completion']
    full_texts = [inp + out for inp, out in zip(inputs, outputs)]

    # Tokenize inputs and full texts
    tokenized_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=512,  # Adjust based on your data
        padding=False
    )
    tokenized_full_texts = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,  # Adjust based on your data
        padding=False
    )

    input_ids = tokenized_full_texts['input_ids']
    labels = []

    for i in range(len(input_ids)):
        input_len = len(tokenized_inputs['input_ids'][i])
        label = input_ids[i].copy()
        # Mask the prompt part in the labels
        label[:input_len] = [-100] * input_len
        labels.append(label)

    tokenized_inputs['input_ids'] = input_ids
    tokenized_inputs['attention_mask'] = tokenized_full_texts['attention_mask']
    tokenized_inputs['labels'] = labels

    return tokenized_inputs

# Apply the preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['prompt', 'completion']
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./fine_tuned_model',
    per_device_train_batch_size=1,   # Adjust based on your GPU memory
    gradient_accumulation_steps=8,   # Simulate larger batch size
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,                       # Enable mixed precision
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,     # Save memory by freeing intermediate activations
    optim='paged_adamw_8bit',        # Use 8-bit optimizer from bitsandbytes
    lr_scheduler_type='cosine',      # Learning rate scheduler
    warmup_steps=100,                # Warm-up steps
    report_to="none"                  # Disable reporting to avoid unnecessary logs
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model('./fine_tuned_model')
