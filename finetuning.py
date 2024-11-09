import json
import os
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
parser = argparse.ArgumentParser(description="Fine-tune Falcon model with optional retraining or interactive mode.")
parser.add_argument(
    "--retrain",
    action="store_true",
    help="If set, retrain the model even if a saved model exists."
)
parser.add_argument(
    "--interactive",
    action="store_true",
    help="If set, load the trained model and start an interactive chat session."
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
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.float16
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
    Loads instruction, input, and output pairs from a JSONL file.
    Combines 'instruction' and 'input' to create 'prompt', and uses 'output' as 'completion'.
    """
    prompts = []
    completions = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            instruction = entry.get('instruction', '')
            input_text = entry.get('input', '')
            output = entry.get('output', '')
            prompt = instruction.strip() + '\n' + input_text.strip()
            prompts.append(prompt)
            completions.append(output)
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
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}%"
    )

# --------------------------
# Function to preprocess data
# --------------------------
def preprocess_function(examples, tokenizer):
    """
    Tokenizes the input prompts and completions.
    Masks the prompt part in the labels to ignore them during loss computation.
    """
    prompts = examples['prompt']
    completions = examples['completion']
    full_texts = [prompt + '\n' + completion for prompt, completion in zip(prompts, completions)]

    # Tokenize prompts
    tokenized_prompts = tokenizer(
        prompts,
        truncation=True,
        max_length=1024,  # Adjust based on your data
        padding='max_length'
    )

    # Tokenize full texts
    tokenized_full_texts = tokenizer(
        full_texts,
        truncation=True,
        max_length=1024,  # Adjust based on your data
        padding='max_length'
    )

    input_ids = tokenized_full_texts['input_ids']
    attention_mask = tokenized_full_texts['attention_mask']
    labels = []

    for i in range(len(input_ids)):
        # Calculate the actual length of the prompt (excluding padding)
        prompt_len = sum(token != tokenizer.pad_token_id for token in tokenized_prompts['input_ids'][i])
        label = input_ids[i].copy()
        # Mask the prompt part in the labels
        label[:prompt_len] = [-100] * prompt_len
        labels.append(label)

    # Prepare the final tokenized inputs
    tokenized_full_texts['input_ids'] = input_ids
    tokenized_full_texts['attention_mask'] = attention_mask
    tokenized_full_texts['labels'] = labels
    # Keep the prompts and completions
    tokenized_full_texts['prompt'] = prompts
    tokenized_full_texts['completion'] = completions

    return tokenized_full_texts

# --------------------------
# Function to evaluate the model
# --------------------------
def evaluate_model(model, tokenizer, eval_dataset, device='cuda', batch_size=1, max_new_tokens=40, num_beams=1):
    """
    Generates completions for the evaluation dataset, extracts unique tool names,
    computes ROUGE metrics, and dumps evaluation details into a JSONL file.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer associated with the model.
        eval_dataset: The evaluation dataset.
        device (str): The device to run the model on ('cuda' or 'cpu').
        batch_size (int): Number of samples per batch.
        max_new_tokens (int): Maximum number of new tokens to generate.
        num_beams (int): Number of beams for beam search.

    Returns:
        result (dict): ROUGE scores.
        generated_completions (list): List of extracted tool names from generated outputs.
        reference_completions (list): List of expected tool names.
        prompts (list): List of prompts used for generation.
    """
    # Load ROUGE metric
    rouge = evaluate.load("rouge")

    # Define the list of possible tools
    possible_tools = [
        "get_func_definition",
        "get_path_constraint",
        "variable_def_finder",
        "get_function_arguments",
        "get_path_constraints",
        "get_buffer_size",
        "get_data_size",
        "get_variable_usage_paths"
    ]

    # Create a regex pattern to match any of the possible tools
    # The pattern ensures full word matching using word boundaries
    tool_pattern = re.compile(r'\b(' + '|'.join(map(re.escape, possible_tools)) + r')\b')

    # Create DataLoader for evaluation
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    # Initialize lists to store results
    generated_completions = []
    reference_completions = []
    prompts = []
    model_outputs = []
    extracted_outputs = []

    model.eval()

    # Generation parameters
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,  # Number of new tokens to generate
        "num_beams": num_beams,
        "early_stopping": True,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id  # Ensure pad_token_id is set
    }

    # Define the output JSONL file
    output_jsonl = 'evaluation_details.jsonl'

    # Remove existing JSONL file if it exists to avoid appending to old data
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Generating completions"):
            prompts_batch = batch['prompt']
            input_ids_list = []
            attention_mask_list = []
            for prompt in prompts_batch:
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding='max_length')
                input_ids_list.append(inputs['input_ids'][0])
                attention_mask_list.append(inputs['attention_mask'][0])

            input_ids = torch.stack(input_ids_list).to(model.device)
            attention_mask = torch.stack(attention_mask_list).to(model.device)

            # Generate completions
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            model_outputs.extend(decoded_preds)

            # Process each generated completion to extract unique tools
            processed_completions = []
            for pred in decoded_preds:
                # Find all tool matches in the prediction
                tools_found = tool_pattern.findall(pred)
                # Get unique tools while preserving order
                unique_tools = list(dict.fromkeys(tools_found))
                # Join with commas
                if unique_tools:
                    processed_completion = ', '.join(unique_tools)
                else:
                    processed_completion = ""  # Assign empty string if no tools found
                processed_completions.append(processed_completion)
            extracted_outputs.extend(processed_completions)
            generated_completions.extend(processed_completions)
            reference_completions.extend(batch['completion'])
            prompts.extend(prompts_batch)

            # Dump the current batch's details to the JSONL file
            with open(output_jsonl, 'a') as f:
                for i in range(len(prompts_batch)):
                    f.write(json.dumps({
                        'prompt': prompts_batch[i],
                        'model_output': decoded_preds[i],
                        'extracted_output': processed_completions[i],
                        'expected_output': batch['completion'][i]
                    }) + '\n')

    # Compute ROUGE scores
    result = rouge.compute(predictions=generated_completions, references=reference_completions, use_stemmer=True)

    # Scale the scores
    result = {key: value * 100 for key, value in result.items()}

    # Round the results for readability
    result = {k: round(v, 4) for k, v in result.items()}

    print(f"Saved evaluation details to '{output_jsonl}'")

    return result, generated_completions, reference_completions, prompts

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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding='max_length').to(model.device)

        # Generate completion
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=128,  # Adjust as needed
                num_beams=1,         # Adjust as needed
                early_stopping=True,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id  # Ensure pad_token_id is set
            )
        # Exclude the prompt tokens from the outputs
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        completion = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

        # Reference completion
        reference_completion = sample['completion']

        print(f"\nSample {i+1}:")
        print("Prompt:")
        print(prompt)
        print("Generated Completion:")
        print(completion)
        print("Reference Completion:")
        print(reference_completion)

# --------------------------
# Function to save evaluation summaries to a JSONL file
# --------------------------
def save_evaluation_summaries(generated_completions, reference_completions, prompts, filename, num_samples=100):
    """
    Saves generated and reference summaries to a JSONL file.
    """
    with open(filename, 'w') as f:
        for i in tqdm(range(min(num_samples, len(generated_completions))), desc=f"Saving summaries to {filename}"):
            f.write(json.dumps({
                'prompt': prompts[i],
                'reference_completion': reference_completions[i],
                'generated_completion': generated_completions[i]
            }) + '\n')
    print(f"Saved {min(num_samples, len(generated_completions))} summaries to {filename}")

# --------------------------
# Main Execution Flow
# --------------------------

def main():
    # --------------------------
    # Handle Interactive Mode
    # --------------------------
    if args.interactive:
        if not os.path.exists(save_directory):
            logger.error(f"Save directory '{save_directory}' does not exist. Cannot enter interactive mode.")
            exit(1)
        logger.info(f"Loading the trained model from '{save_directory}' for interactive chat.")

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            "tiiuae/falcon-40b-instruct",
            cache_dir=cache_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            # quantization_config=bnb_config,
            trust_remote_code=False
        )

        # Load the LoRA adapters using PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, save_directory)
        model.eval()
        model.config.use_cache = True

        # Load the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(
            "tiiuae/falcon-40b-instruct",
            cache_dir=cache_dir,
            padding_side='left'
        )

        # Set the pad_token to eos_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Prepare the model for inference
        print("\nEnter 'exit' to quit the chat.")
        while True:
            user_input = input("User: ")
            if user_input.lower() == 'exit':
                break
            if user_input.strip() == "":
                print("Please enter a valid prompt or type 'exit' to quit.")
                continue
            # Tokenize the user input
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512, padding='max_length').to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=128,
                    num_beams=1,
                    early_stopping=True,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=tokenizer.eos_token_id
                )
            # Exclude the prompt tokens from the outputs
            generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
            print(f"Model: {response}")
        exit(0)  # Exit the script after the chat session

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
    # Dump prompt and completion pairs to JSONL files
    # --------------------------
    dump_prompt_completion(train_dataset, 'train_prompt_completion.jsonl')
    dump_prompt_completion(eval_dataset, 'eval_prompt_completion.jsonl')

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
    # Apply the preprocessing
    # --------------------------
    tokenized_train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names  # Remove original columns
    )

    tokenized_eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names  # Remove original columns
    )

    # --------------------------
    # Set format for PyTorch tensors, include prompt and completion
    # --------------------------
    tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    tokenized_eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)

    # --------------------------
    # Check if a saved model exists and load it if not retraining
    # --------------------------
    if os.path.exists(save_directory) and not args.retrain:
        logger.info(f"Found a saved model in '{save_directory}'. Loading the model and skipping training.")

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=False
        )

        # Load the LoRA adapters using PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, save_directory)
        model.eval()
        model.config.use_cache = True

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
        # Define training arguments
        # --------------------------
        training_args = TrainingArguments(
            output_dir='./fine_tuned_model',             # Directory to save the model checkpoints
            per_device_train_batch_size=1,              # Keep batch size small due to potential GPU memory constraints
            per_device_eval_batch_size=1,               # Same as training batch size
            gradient_accumulation_steps=8,              # Increase to simulate a larger effective batch size
            num_train_epochs=30,                        # Significantly increase epochs to allow extensive training
            learning_rate=1e-5,                          # Lower learning rate for finer weight updates
            weight_decay=0.0,                            # Remove weight decay to reduce regularization
            logging_dir='./logs',                        # Directory for logging
            logging_steps=10,                            # Increase logging frequency for better monitoring
            save_strategy="steps",                       # Save checkpoints based on steps
            save_steps=100,                              # Save every 100 steps to capture more checkpoints
            save_total_limit=None,                       # Allow unlimited checkpoints to prevent overwriting
            evaluation_strategy="steps",                    # Disable evaluation to focus solely on training accuracy
            load_best_model_at_end=True,                # Do not load the best model based on evaluation (since eval is disabled)
            # metric_for_best_model="eval_loss",         # Not needed as evaluation is disabled
            fp16=True,                                   # Use mixed precision for faster training
            optim="adamw_torch",                         # Use AdamW optimizer
            lr_scheduler_type="linear",                  # Use a linear scheduler for simplicity
            warmup_steps=100,                            # Minimal warmup steps to stabilize training start
            max_grad_norm=0.0,                            # Disable gradient clipping to allow larger gradient updates
            gradient_checkpointing=True,                 # Enable gradient checkpointing to save memory
            torch_compile=False,                         # Disable Torch compilation for compatibility
            report_to="none",                            # Disable reporting to external systems
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
        # Save the LoRA adapters
        # --------------------------
        os.makedirs(save_directory, exist_ok=True)
        model.save_pretrained(save_directory)
        print(f"Model saved to '{save_directory}'")

    # --------------------------
    # Proceed with Evaluation (Only if Not in Interactive Mode)
    # --------------------------
    if not args.interactive:
        logger.info("Starting evaluation...")

        # Perform Evaluation with Generation and Compute ROUGE Metrics
        evaluation_results, generated_completions, reference_completions, prompts = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=tokenized_eval_dataset,
            device='cuda',            # Change to 'cpu' if GPU is not available
            batch_size=1,             # Adjust batch size as needed
            max_new_tokens=128,       # Adjust max_new_tokens as needed
            num_beams=1               # Adjust num_beams as needed
        )

        print("\nEvaluation Results (ROUGE scores):")
        print(evaluation_results)

        # Show ground truth and generated outputs
        print("\nGround Truth vs Generated Completions:")
        num_samples_to_show = 5
        for i in range(num_samples_to_show):
            print(f"\nSample {i+1}:")
            print("Prompt:")
            print(prompts[i])
            print("Ground Truth Completion:")
            print(reference_completions[i])
            print("Generated Completion:")
            print(generated_completions[i])

        # Optional: Generate and Print Sample Summaries
        generate_sample_summaries(
            eval_dataset=tokenized_eval_dataset,  # Use the tokenized eval_dataset
            tokenizer=tokenizer,
            model=model,
            device='cuda',    # Change to 'cpu' if GPU is not available
            num_samples=5
        )

        # Save evaluation summaries to a JSONL file
        save_evaluation_summaries(
            generated_completions=generated_completions,
            reference_completions=reference_completions,
            prompts=prompts,
            filename='evaluation_summaries.jsonl',
            num_samples=100
        )

if __name__ == "__main__":
    main()
