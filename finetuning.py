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
# Run in terminal:
# pip install --upgrade bitsandbytes

# Specify the cache directory
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/falcon'

# Define quantization configuration using BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16,  # Ensure computations are in float16
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
    torch_dtype=torch.float32,  # Set to float32 to prevent dtype mismatch
    quantization_config=quantization_config,
    trust_remote_code=False
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
for name, param in model.named_parameters():
    if 'lora_' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

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
        # Calculate the actual length of the prompt (excluding padding)
        input_len = sum(token != tokenizer.pad_token_id for token in tokenized_inputs['input_ids'][i])
        label = input_ids[i].copy()
        # Mask the prompt part in the labels
        label[:input_len] = [-100] * input_len
        labels.append(label)

    tokenized_full_texts['labels'] = labels

    return tokenized_full_texts

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

# **Training arguments with adjustments based on reference code**
training_args = TrainingArguments(
    output_dir='./fine_tuned_model',
    per_device_train_batch_size=1,       # Adjust based on your GPU memory
    gradient_accumulation_steps=8,       # Simulate larger batch size
    num_train_epochs=3,
    learning_rate=3e-5,                  # Adjusted learning rate for stability
    fp16=False,                          # Disable mixed precision to prevent dtype issues
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    gradient_checkpointing=True,         # Save memory by freeing intermediate activations
    optim="adamw_torch",                 # Use standard AdamW optimizer
    lr_scheduler_type='cosine_with_restarts',  # Changed scheduler for better convergence
    warmup_ratio=0.05,                   # Adjusted warmup steps
    report_to="none",                    # Disable reporting to avoid unnecessary logs
    evaluation_strategy="steps",         # Enable evaluation during training
    eval_steps=100,                      # Evaluate every 100 steps
    save_strategy="steps",
    load_best_model_at_end=True,         # Load best model based on evaluation metric
    metric_for_best_model="loss",        # Use loss as evaluation metric
)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Not using masked language modeling
)

# **Initialize the Trainer with evaluation and callbacks**
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # Using the same dataset for evaluation; replace with a valid eval dataset
    data_collator=data_collator,
    callbacks=[]  # Add any custom callbacks if needed
)

# **Start training**
trainer.train()

# **Save the final model**
trainer.save_model('./fine_tuned_model')
