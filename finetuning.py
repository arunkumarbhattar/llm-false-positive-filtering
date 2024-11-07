import json
import os
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig, TaskType
from transformers import TrainerCallback, EarlyStoppingCallback
import logging
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader

# --------------------------
# Set up logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Parse command-line arguments
# --------------------------
parser = argparse.ArgumentParser(description="Fine-tune Falcon model with optional retraining.")
parser.add_argument(
    "--retrain",
    action="store_true",
    help="If set, retrain the model even if a saved model exists."
)
args = parser.parse_args()

# --------------------------
# Specify the cache directory
# --------------------------
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/falcon'

# --------------------------
# Define quantization configuration using BitsAndBytesConfig for 4-bit QLoRA
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16  # Can be float16 or bfloat16
)

# --------------------------
# Define save directory
# --------------------------
save_directory = '/scratch/gilbreth/bhattar1/transformers/saved_falcon_codeql'

# --------------------------
# Function to load data from JSONL file
# --------------------------
def load_jsonl(file_path):
    """
    Loads prompt and completion pairs from a JSONL file.
    Each line in the file should be a JSON object with 'prompt' and 'completion' keys.
    """
    prompts = []
    completions = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompts.append(entry['prompt'])
            completions.append(entry['completion'])
    return {'prompt': prompts, 'completion': completions}

# --------------------------
# Function to dump prompt and completion pairs to JSONL files
# --------------------------
def dump_prompt_completion(dataset, filename):
    """
    Dumps the prompt and completion pairs to a JSONL file.
    """
    # Reload the original dataset before mapping to get exact prompts and completions
    original_prompts = dataset['prompt']
    original_completions = dataset['completion']

    with open(filename, 'w') as f:
        for prompt, completion in zip(original_prompts, original_completions):
            f.write(json.dumps({
                'prompt': prompt,
                'completion': completion
            }) + '\n')
    print(f"Dumped {len(original_prompts)} samples to {filename}")

# --------------------------
# Function to print trainable parameters
# --------------------------
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )

# --------------------------
# Function to preprocess data
# --------------------------
def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input prompts and completions.
    Masks the prompt part in the labels to ignore them during loss computation.
    """
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
    attention_mask = tokenized_full_texts['attention_mask']
    labels = []

    for i in range(len(input_ids)):
        # Calculate the actual length of the prompt (excluding padding)
        input_len = sum(token != tokenizer.pad_token_id for token in tokenized_inputs['input_ids'][i])
        label = input_ids[i].copy()
        # Mask the prompt part in the labels
        label[:input_len] = [-100] * input_len
        labels.append(label)

    # Prepare the final tokenized inputs
    tokenized_full_texts['labels'] = labels

    return tokenized_full_texts

# --------------------------
# Function to evaluate the model
# --------------------------
def evaluate_model(model, tokenizer, eval_dataset, device='cuda', batch_size=1, max_new_tokens=128, num_beams=1):
    """
    Generates summaries for the evaluation dataset and computes ROUGE metrics.
    Also shows ground truth and generated outputs.
    """
    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    # Create DataLoader for evaluation
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    generated_summaries = []
    reference_summaries = []
    prompts = []

    model.eval()
    # model.to(device)  # Removed: Moving the model to a device is not supported for 4-bit quantized models

    # Modified Generation Parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,  # Specify the number of new tokens to generate
        "num_beams": num_beams,
        "early_stopping": True,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id  # Ensure pad_token_id is set
    }

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating summaries"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Generate summaries
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generation_kwargs)
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Decode references
            labels = batch['labels']
            # Replace -100 in labels as we set them to ignore in loss computation
            labels = [
                [l if l != -100 else tokenizer.pad_token_id for l in label] for label in labels
            ]
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Append to lists
            generated_summaries.extend(decoded_preds)
            reference_summaries.extend(decoded_labels)
            # Also collect prompts
            batch_prompts = tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=True)
            prompts.extend(batch_prompts)

    # Compute ROUGE scores
    result = rouge.compute(predictions=generated_summaries, references=reference_summaries, use_stemmer=True)

    # Scale the scores
    result = {key: value * 100 for key, value in result.items()}

    # Round the results for readability
    result = {k: round(v, 4) for k, v in result.items()}

    return result, generated_summaries, reference_summaries, prompts

# --------------------------
# Function to generate and print sample summaries
# --------------------------
def generate_sample_summaries(eval_dataset, tokenizer, model, device='cuda', num_samples=5):
    """
    Generates and prints sample summaries from the evaluation dataset.
    """
    print("\nSample Generated Summaries:")
    model.eval()
    for i in range(num_samples):
        sample = eval_dataset[i]
        prompt = sample['prompt']
        # Tokenize the prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        attention_mask = tokenizer(prompt, return_tensors="pt").attention_mask.to(device)

        # Generate summary
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,  # Reduced max_new_tokens
                num_beams=1,         # Reduced num_beams
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set
            )
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Reference summary
        # Replace -100 in labels to get the actual token IDs
        labels = sample['labels']
        labels = [l if l != -100 else tokenizer.pad_token_id for l in labels]
        reference_summary = tokenizer.decode(labels, skip_special_tokens=True)

        print(f"\nSample {i+1}:")
        print("Prompt:")
        print(prompt)
        print("Generated Summary:")
        print(summary)
        print("Reference Summary:")
        print(reference_summary)

# --------------------------
# Function to save evaluation summaries to a JSONL file
# --------------------------
def save_evaluation_summaries(generated_summaries, reference_summaries, prompts, filename, num_samples=100):
    """
    Saves generated and reference summaries to a JSONL file.
    """
    with open(filename, 'w') as f:
        for i in tqdm(range(min(num_samples, len(generated_summaries))), desc=f"Saving summaries to {filename}"):
            f.write(json.dumps({
                'prompt': prompts[i],
                'reference_summary': reference_summaries[i],
                'generated_summary': generated_summaries[i]
            }) + '\n')
    print(f"Saved {min(num_samples, len(generated_summaries))} summaries to {filename}")

# --------------------------
# Load the tokenizer
# --------------------------
model_id = 'tiiuae/falcon-40b-instruct'
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    padding_side='left'  # Ensure padding side is left
)

# Set the pad_token to eos_token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --------------------------
# Load your training data
# --------------------------
data = load_jsonl('fine_tuning_training_data.jsonl')
dataset = Dataset.from_dict(data)

# --------------------------
# Split the dataset into training and evaluation sets
# --------------------------
split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# --------------------------
# Dump prompt and answer pairs to JSONL files
# --------------------------
dump_prompt_completion(train_dataset, 'train_prompt_completion.jsonl')
dump_prompt_completion(eval_dataset, 'eval_prompt_completion.jsonl')

# --------------------------
# Apply the preprocessing
# --------------------------
tokenized_train_dataset = train_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=['prompt', 'completion']
)

tokenized_eval_dataset = eval_dataset.map(
    lambda examples: preprocess_function(examples, tokenizer),
    batched=True,
    remove_columns=['prompt', 'completion']
)

# --------------------------
# Set format for PyTorch tensors
# --------------------------
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# --------------------------
# Check if a saved model exists and load it if not retraining
# --------------------------
if os.path.exists(save_directory) and not args.retrain:
    logger.info(f"Found a saved model in {save_directory}. Loading the model and skipping training.")

    # # Load the model from the save_directory with quantization_config
    # model = AutoModelForCausalLM.from_pretrained(
    #     save_directory,
    #     cache_dir=cache_dir,
    #     device_map='auto',
    #     torch_dtype=torch.float16,
    #     quantization_config=bnb_config,
    #     trust_remote_code=False
    # )

else:
    logger.info("No saved model found or retraining requested. Proceeding with training.")

    # Load the base model with quantization_config
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map='auto',
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        trust_remote_code=False
    )

    # Set use_cache to False to avoid incompatibility with gradient checkpointing
    model.config.use_cache = False

    # Set pad_token_id in the model's configuration
    model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Prepare the model for k-bit (4-bit) training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Apply LoRA configurations
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Apply PEFT model
    model = get_peft_model(model, peft_config)

    # Verify trainable parameters
    print_trainable_parameters(model)

    # --------------------------
    # Define training arguments based on reference code
    # --------------------------
    training_args = TrainingArguments(
        output_dir='./fine_tuned_model',
        per_device_train_batch_size=1,          # Reduced batch size
        per_device_eval_batch_size=1,           # Reduced eval batch size
        gradient_accumulation_steps=4,          # Adjusted to simulate larger batch size without exceeding GPU memory
        num_train_epochs=3,                     # Adjusted epochs
        learning_rate=1e-4,                     # Learning rate
        weight_decay=0.01,                      # Weight decay
        logging_dir='./logs',                   # Local logging dir
        logging_steps=50,                       # Logging steps
        save_strategy="steps",
        save_steps=500,                         # Save steps
        save_total_limit=3,                     # Total limit
        evaluation_strategy="steps",
        eval_steps=500,                         # Eval steps
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,                              # Enable mixed precision
        optim="adamw_torch",                    # Optimizer
        lr_scheduler_type="cosine_with_restarts",
        warmup_ratio=0.1,                       # Warmup ratio
        max_grad_norm=1.0,                      # Max grad norm
        gradient_checkpointing=True,            # Gradient checkpointing
        torch_compile=False,
        report_to="none",                       # Disabled reporting
    )

    # --------------------------
    # Define custom callback for detecting abnormal loss
    # --------------------------
    class CustomCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # Ensure that log_history is non-empty before accessing the last element
            if not state.log_history:
                logger.warning(f"At step {state.global_step}: 'log_history' is empty.")
                return  # Exit early since there's nothing to process

            # Access the last entry in log_history
            last_log = state.log_history[-1]

            # Check if 'loss' is present in the last log entry
            if 'loss' in last_log:
                loss = last_log['loss']
                logger.debug(f"At step {state.global_step}: loss = {loss}")

                # Detect abnormal loss values
                if loss > 1e4 or loss < 0:
                    logger.error(f"Abnormal loss detected at step {state.global_step}: {loss}")
                    control.should_terminate_training = True
            else:
                logger.debug(f"At step {state.global_step}: 'loss' not found in log_history.")

    # --------------------------
    # Data collator for language modeling
    # --------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Not using masked language modeling
    )

    # --------------------------
    # Initialize the Trainer
    # --------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), CustomCallback()],
        tokenizer=tokenizer,  # This ensures the tokenizer is saved with the model
    )

    # --------------------------
    # Start training
    # --------------------------
    trainer.train()

    # --------------------------
    # Save the final model to the save_directory
    # --------------------------
    os.makedirs(save_directory, exist_ok=True)
    trainer.save_model(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model saved to {save_directory}")


# Load the base model with quantization_config
model = AutoModelForCausalLM.from_pretrained(
    save_directory,
    cache_dir=cache_dir,
    device_map='auto',
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=False
)

# Prepare the model for inference
model.eval()
model.config.use_cache = True  # Enable cache for inference

# --------------------------
# Perform Evaluation with Generation and Compute ROUGE Metrics
# --------------------------
evaluation_results, generated_summaries, reference_summaries, prompts = evaluate_model(
    model=model,
    tokenizer=tokenizer,
    eval_dataset=tokenized_eval_dataset,
    device='cuda',            # Change to 'cpu' if GPU is not available
    batch_size=1,             # Reduced batch size
    max_new_tokens=128,       # Reduced max_new_tokens
    num_beams=1               # Reduced num_beams
)

print("Evaluation Results (ROUGE scores):")
print(evaluation_results)

# --------------------------
# Show ground truth and generated outputs
# --------------------------
print("\nGround Truth vs Generated Summaries:")
num_samples_to_show = 5
for i in range(num_samples_to_show):
    print(f"\nSample {i+1}:")
    print("Prompt:")
    print(prompts[i])
    print("Ground Truth:")
    print(reference_summaries[i])
    print("Generated Summary:")
    print(generated_summaries[i])

# --------------------------
# Optional: Generate and Print Sample Summaries
# --------------------------
generate_sample_summaries(
    eval_dataset=tokenized_eval_dataset,
    tokenizer=tokenizer,
    model=model,
    device='cuda',    # Change to 'cpu' if GPU is not available
    num_samples=5
)

# --------------------------
# Save evaluation summaries to a JSONL file
# --------------------------
save_evaluation_summaries(
    generated_summaries=generated_summaries,
    reference_summaries=reference_summaries,
    prompts=prompts,
    filename='evaluation_summaries.jsonl',
    num_samples=100
)

# --------------------------
# Chat prompt for the user
# --------------------------
print("\nEnter 'exit' to quit the chat.")
model.eval()
while True:
    user_input = input("User: ")
    if user_input.lower() == 'exit':
        break
    # Tokenize the user input
    inputs = tokenizer(user_input, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Reduced max_new_tokens
            num_beams=1,         # Reduced num_beams
            early_stopping=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Model: {response}")
