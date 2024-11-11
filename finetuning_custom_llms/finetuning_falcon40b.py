import json
import os
import random
import re
import torch
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)
import logging

# --------------------------
# Set up logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Specify the cache directory
# --------------------------
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/falcon'

# --------------------------
# Define quantization configuration using BitsAndBytesConfig for 8-bit QLoRA
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True
)

# --------------------------
# Define save directory
# --------------------------
save_directory = '/scratch/gilbreth/bhattar1/transformers/saved_falcon_codeql'

# --------------------------
# Function to load JSONL data with reasoning
# --------------------------
def load_jsonl_with_reasoning(file_path):
    """
    Loads 'prompt' and 'completion' from a JSONL file.

    Args:
        file_path (str): Path to the JSONL file.

    Returns:
        dict: A dictionary with 'prompt' and 'completion' lists.
    """
    prompts = []
    completions = []
    with open(file_path, 'r') as f:
        for idx, line in enumerate(f):
            try:
                entry = json.loads(line)

                # Extract fields with default empty strings if not present
                instruction = entry.get('instruction', '').strip()
                input_text = entry.get('input', '').strip()
                output = entry.get('output', '').strip()

                # Combine 'instruction' and 'input' to form 'prompt'
                prompt = f"{instruction}\n{input_text}"
                prompts.append(prompt)

                # Use 'output' as 'completion'
                completions.append(output)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decoding failed at line {idx+1}: {e}")
                prompts.append('')
                completions.append('')
    return {'prompt': prompts, 'completion': completions}

# --------------------------
# Function to dump prompt and completion pairs to JSONL files
# --------------------------
def dump_prompt_completion(dataset, filename):
    """
    Dumps the prompt and completion pairs to a JSONL file.
    """
    prompts = dataset['prompt']
    completions = dataset['completion']

    with open(filename, 'w') as f:
        for prompt, completion in zip(prompts, completions):
            f.write(json.dumps({
                'prompt': prompt,
                'completion': completion
            }) + '\n')
    print(f"Dumped {len(prompts)} samples to {filename}")

# --------------------------
# Function to preprocess data
# --------------------------
def preprocess_function(examples, tokenizer):
    """
    Preprocesses the input examples by tokenizing the prompts and completions.

    Args:
        examples (dict): A batch of examples containing 'prompt' and 'completion'.
        tokenizer: The tokenizer associated with the model.

    Returns:
        dict: A dictionary containing tokenized inputs and labels, along with the original 'prompt' and 'completion'.
    """
    inputs = examples['prompt']
    targets = examples['completion']

    # Tokenize the prompts
    model_inputs = tokenizer(
        inputs,
        max_length=1024,
        truncation=True,
        padding='max_length'
    )

    # Tokenize the completions as labels
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=128,
            truncation=True,
            padding='max_length'
        )

    model_inputs['labels'] = labels['input_ids']

    # Explicitly preserve 'prompt' and 'completion' fields
    model_inputs['prompt'] = examples['prompt']
    model_inputs['completion'] = examples['completion']

    return model_inputs

# --------------------------
# Parse command-line arguments
# --------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Falcon model with LoRA adapters.")

    # Existing arguments
    parser.add_argument('--interactive', action='store_true', help='Enter interactive chat mode after loading the model.')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model even if a saved model exists.')
    parser.add_argument('--only_eval', action='store_true', help='Perform only evaluation without training.')
    parser.add_argument('--train', action='store_true', help='Perform training.')

    # Parse the arguments
    args = parser.parse_args()
    return args

# --------------------------
# Main Execution Flow
# --------------------------

def main():
    args = parse_arguments()

    if args.only_eval:
        # Evaluation code
        exit(0)

    elif args.interactive:
        # Interactive code
        exit(0)

    else:
        # Proceed to training
        print("Proceeding to training...")

        # --------------------------
        # Load your training data
        # --------------------------
        data = load_jsonl_with_reasoning('../prompt_pair_prepping/fine_tuning_training_data.jsonl')
        dataset = Dataset.from_dict(data)

        # --------------------------
        # Split the dataset into training and evaluation sets
        # --------------------------
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        # --------------------------
        # Sample 15% of the training data
        # --------------------------
        train_dataset = train_dataset.shuffle(seed=42).select(range(int(0.15 * len(train_dataset))))

        # Sample 15% of the evaluation data (optional)
        eval_dataset = eval_dataset.shuffle(seed=42).select(range(int(0.15 * len(eval_dataset))))

        # --------------------------
        # Dump prompt and completion pairs to JSONL files
        # --------------------------
        dump_prompt_completion(train_dataset, 'train_prompt_completion_falcon.jsonl')
        dump_prompt_completion(eval_dataset, 'eval_prompt_completion_falcon.jsonl')

        # --------------------------
        # Load the tokenizer
        # --------------------------
        model_id = 'tiiuae/falcon-40b-instruct'
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side='left'
        )

        # Set the pad_token to eos_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --------------------------
        # Apply the preprocessing
        # --------------------------
        tokenized_train_dataset = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=[],
            load_from_cache_file=False
        )

        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=[],
            load_from_cache_file=False
        )

        # Set format for PyTorch tensors
        tokenized_train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True
        )
        tokenized_eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True
        )

        # --------------------------
        # Load the base model with quantization_config
        # --------------------------
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

        # Enable gradient checkpointing
        model.gradient_checkpointing_enable()

        # Prepare the model for k-bit (8-bit) training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # --------------------------
        # Define the target module names specific to Falcon
        # --------------------------
        target_module_names = [
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ]

        # --------------------------
        # Apply LoRA configurations with correct target modules
        # --------------------------
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=target_module_names,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Apply PEFT model
        model = get_peft_model(model, peft_config)

        # --------------------------
        # Define training arguments with adjustments for faster training
        # --------------------------
        training_args = TrainingArguments(
            output_dir='./fine_tuned_model',
            per_device_train_batch_size=1,   # Smaller batch size to fit in memory
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,   # No accumulation for simplicity
            num_train_epochs=2,              # Only 1 epoch for quick training
            learning_rate=1e-4,              # Slightly higher learning rate
            logging_steps=1000,              # Less frequent logging
            save_strategy="no",              # Disable checkpoint saving
            evaluation_strategy="no",        # Disable evaluation during training
            fp16=True,                       # Use mixed precision
            optim="adamw_torch",             # Optimizer
            lr_scheduler_type="linear",      # Learning rate scheduler
            warmup_steps=0,                  # No warmup
            report_to="none",                # Disable reporting
        )

        # --------------------------
        # Data collator for language modeling
        # --------------------------
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )

        # --------------------------
        # Initialize the Trainer
        # --------------------------
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # --------------------------
        # Start training
        # --------------------------
        trainer.train()

        # --------------------------
        # Save the LoRA adapters
        # --------------------------
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        print(f"Model saved to '{save_directory}'")

if __name__ == "__main__":
    main()
