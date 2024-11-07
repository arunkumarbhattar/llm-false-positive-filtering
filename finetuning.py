import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType

# Ensure bitsandbytes is up-to-date
# You should run the following command in your terminal:
# pip install --upgrade bitsandbytes

# Specify the cache directory
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/falcon'

# Define quantization configuration using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Example parameter; adjust based on your needs
    llm_int8_has_fp16_weight=True  # Example parameter; adjust based on your needs
)

# Load the tokenizer without trust_remote_code
model_name = 'tiiuae/falcon-40b-instruct'

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

# **Set the pad_token to eos_token**
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the model with quantization_config and without trust_remote_code
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    device_map='auto',
    torch_dtype=torch.float16,
    quantization_config=quantization_config,
    trust_remote_code=False  # Removed as per your instruction
)

# **Set use_cache to False to avoid incompatibility with gradient checkpointing**
model.config.use_cache = False

# **Set pad_token_id in the model's configuration**
model.config.pad_token_id = tokenizer.eos_token_id

# **Apply LoRA for efficient fine-tuning**
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,                # Rank of the update matrices
    lora_alpha=32,       # Scaling factor
    lora_dropout=0.05    # Dropout for LoRA layers
)

model = get_peft_model(model, peft_config)

# **Freeze base model parameters to ensure only LoRA parameters are trainable**
for param in model.base_model.parameters():
    param.requires_grad = False

# **Ensure LoRA parameters are trainable**
for param in model.parameters():
    if not param.requires_grad:
        param.requires_grad = True

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

    # Tokenize prompts
    tokenized_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=512,  # Adjust based on your data
        padding='max_length'
    )

    # Tokenize full texts
    tokenized_full_texts = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,  # Adjust based on your data
        padding='max_length'
    )

    input_ids = tokenized_full_texts['input_ids']
    labels = []

    for i in range(len(input_ids)):
        input_len = len([token for token in tokenized_inputs['input_ids'][i] if token != tokenizer.pad_token_id])
        label = input_ids[i].copy()
        # Mask the prompt part in the labels
        label[:input_len] = [-100] * input_len
        labels.append(label)

    tokenized_inputs['labels'] = labels

    return tokenized_inputs

# Apply the preprocessing
tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=['prompt', 'completion']
)

# Set format for PyTorch tensors
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# **Verify that there are trainable parameters**
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
if trainable_params == 0:
    raise ValueError("No trainable parameters found. Check if PEFT is applied correctly.")

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
    optim="paged_adamw_8bit",        # Use 8-bit optimizer from bitsandbytes
    lr_scheduler_type='cosine',      # Learning rate scheduler
    warmup_steps=100,                # Warm-up steps
    report_to="none"                 # Disable reporting to avoid unnecessary logs
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

# Initialize the Trainer without passing the tokenizer (to avoid deprecation warning)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
trainer.train()

# Save the final model
trainer.save_model('./fine_tuned_model')
