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
from transformers import TrainerCallback, EarlyStoppingCallback
import logging
from tqdm import tqdm
import evaluate
from torch.utils.data import DataLoader, Subset

# --------------------------
# Set up logging
# --------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------------------------
# Specify the cache directory
# --------------------------
cache_dir = '/scratch/gilbreth/bhattar1/.cache/huggingface/transformers/mistral'
access_token = "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"
# --------------------------
# Define quantization configuration using BitsAndBytesConfig for 4-bit QLoRA
# --------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                       # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,          # Use double quantization for better accuracy
    bnb_4bit_quant_type="nf4",               # Quantization type; "nf4" is recommended for transformers
    bnb_4bit_compute_dtype=torch.float16      # Compute dtype for 4-bit weights
)

# --------------------------
# Define save directory
# --------------------------
save_directory = '/scratch/gilbreth/bhattar1/transformers/saved_mistral_codeql'

# --------------------------
def load_jsonl_with_reasoning(file_path):
    """
    Loads 'prompt' and 'completion' from a JSONL file.

    Assumptions:
    - Each JSON line contains 'instruction', 'input', and 'output'.
    - 'output' contains '**Tool Selection**: ...\n\n**Reasoning**: ...'

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
# Function to load evaluation data from JSONL file
# --------------------------
def load_eval_jsonl(file_path):
    """
    Loads 'prompt' and 'completion' pairs from a JSONL file.
    """
    prompts = []
    completions = []
    with open(file_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            prompt = entry.get('prompt', '')
            completion = entry.get('completion', '')
            prompts.append(prompt.strip())
            completions.append(completion.strip())
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
    Preprocesses the input examples by tokenizing the prompts and completions.

    Args:
        examples (dict): A batch of examples containing 'prompt' and 'completion'.
        tokenizer: The tokenizer associated with the model.

    Returns:
        dict: A dictionary containing tokenized inputs and labels, along with the original 'prompt' and 'completion'.
    """
    inputs = examples['prompt']
    targets = examples['completion']

    # Combine prompts and completions with a separator
    full_texts = [f"{prompt}\n\n### Answer:\n{completion}" for prompt, completion in zip(inputs, targets)]

    # Tokenize the combined texts
    model_inputs = tokenizer(
        full_texts,
        max_length=1024,
        truncation=True,
        padding='max_length'
    )

    labels = model_inputs['input_ids'].copy()

    model_inputs['labels'] = labels

    # Explicitly preserve 'prompt' and 'completion' fields
    model_inputs['prompt'] = examples['prompt']
    model_inputs['completion'] = examples['completion']

    return model_inputs

# --------------------------
# Custom collate function to handle mixed data types
# --------------------------
def custom_collate_fn(batch):
    """
    Custom collate function to handle mixed data types in batches.

    Args:
        batch (list): A list of dictionaries from the dataset.

    Returns:
        dict: A dictionary with batched tensors and lists of strings.
    """
    # Stack tensor fields
    input_ids = torch.stack([torch.tensor(item['input_ids']) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask']) for item in batch])
    labels = torch.stack([torch.tensor(item['labels']) for item in batch])

    # Collect string fields into lists
    prompts = [item['prompt'] for item in batch]
    completions = [item['completion'] for item in batch]

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'prompt': prompts,
        'completion': completions
    }


# --------------------------
# Function to evaluate the model
# --------------------------
def evaluate_model(model, tokenizer, eval_dataset, device='cuda', batch_size=1, max_new_tokens=40, num_beams=1, num_samples=80, seed=None):
    """
    Generates completions for a randomly sampled subset of the evaluation dataset,
    extracts unique tool names, compares them against expected tools, computes evaluation metrics,
    and dumps evaluation details into a JSONL file.

    Args:
        model: The language model to be evaluated.
        tokenizer: The tokenizer associated with the model.
        eval_dataset: The evaluation dataset containing 'prompt' and 'completion' fields.
        device (str): The device to run the model on ('cuda' or 'cpu').
        batch_size (int): The batch size for evaluation.
        max_new_tokens (int): The maximum number of new tokens to generate.
        num_beams (int): The number of beams for beam search.
        num_samples (int): The number of samples to evaluate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing evaluation metrics.
        list: Generated completions.
        list: Reference completions.
        list: Prompts.
    """

    # --------------------------
    # Set random seed for reproducibility if provided
    # --------------------------
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    # --------------------------
    # Load ROUGE metric
    # --------------------------
    rouge = evaluate.load("rouge")

    # --------------------------
    # Define the list of possible tools (without 'functions.' prefix)
    # --------------------------
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

    # --------------------------
    # Create a regex pattern to match any of the possible tools, possibly preceded by 'functions.'
    # This pattern captures the tool name without the 'functions.' prefix.
    # Example matches: 'functions.get_func_definition' or 'get_func_definition'
    # The tool name is captured in group(1).
    # --------------------------
    tool_pattern = re.compile(r'\b(?:functions\.)?(' + '|'.join(map(re.escape, possible_tools)) + r')\b')

    # --------------------------
    # Determine the number of samples to evaluate
    # --------------------------
    total_samples = len(eval_dataset)
    if num_samples > total_samples:
        raise ValueError(f"num_samples ({num_samples}) cannot be greater than the size of eval_dataset ({total_samples}).")

    # --------------------------
    # Randomly sample indices for evaluation
    # --------------------------
    sampled_indices = random.sample(range(total_samples), num_samples)

    # --------------------------
    # Create a subset of the evaluation dataset
    # --------------------------
    eval_subset = Subset(eval_dataset, sampled_indices)

    print("Columns in eval_dataset:", eval_dataset.column_names)
    print("Columns in eval_subset:", eval_subset.dataset.column_names)  # eval_subset is a Subset

    # --------------------------
    # Create DataLoader for the subset
    # --------------------------
    eval_dataloader = DataLoader(
        eval_subset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn  # Ensure this function is defined elsewhere in your script
    )

    # --------------------------
    # Initialize lists to store results
    # --------------------------
    generated_completions = []
    reference_completions = []
    prompts = []
    model_outputs = []
    extracted_outputs = []
    extracted_expected_tools = []
    extracted_predicted_tools = []

    # --------------------------
    # Move model to the specified device and set to evaluation mode
    # --------------------------
    model.to(device)
    model.eval()

    # --------------------------
    # Define generation parameters
    # --------------------------
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "early_stopping": num_beams > 1,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id
    }

    # --------------------------
    # Define the output JSONL file
    # --------------------------
    output_jsonl = 'evaluation_details_mistral.jsonl'

    # --------------------------
    # Remove existing JSONL file if it exists to avoid appending to old data
    # --------------------------
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    # --------------------------
    # Log the start of evaluation
    # --------------------------
    print("INFO:__main__:Starting evaluation on a subset of the dataset...")

    # --------------------------
    # Initialize counters for set-based metrics
    # --------------------------
    total_correct = 0
    total_predicted = 0
    total_reference = 0

    # --------------------------
    # Initialize lists for set-based metrics
    # --------------------------
    all_predicted_tool_sets = []
    all_reference_tool_sets = []

    # --------------------------
    # Disable gradient computation for evaluation
    # --------------------------
    with torch.no_grad():
        # --------------------------
        # Iterate over batches in the DataLoader
        # --------------------------
        for batch in tqdm(eval_dataloader, desc="Generating completions"):
            prompts_batch = batch['prompt']
            reference_batch = batch['completion']

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # --------------------------
            # Generate completions using the model
            # --------------------------
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            model_outputs.extend(decoded_preds)

            # --------------------------
            # Process each generated completion to extract unique tools
            # --------------------------
            processed_completions = []
            predicted_tool_sets = []
            for pred in decoded_preds:
                # Find all tool matches in the prediction
                tools_found = tool_pattern.findall(pred)
                # Get unique tools while preserving order
                unique_tools = list(dict.fromkeys(tools_found))
                # Extract function names without the 'functions.' prefix
                extracted_tools = [tool for tool in unique_tools]
                predicted_tool_sets.append(set(extracted_tools))
                # Join with commas for logging or other purposes
                if unique_tools:
                    processed_completion = ', '.join(unique_tools)
                else:
                    processed_completion = ""
                processed_completions.append(processed_completion)
            extracted_outputs.extend(processed_completions)
            generated_completions.extend(processed_completions)
            reference_completions.extend(reference_batch)
            prompts.extend(prompts_batch)
            extracted_predicted_tools.extend(predicted_tool_sets)

            # --------------------------
            # Extract expected tools from reference_batch by parsing the tool selection
            # --------------------------
            expected_tool_sets = []
            for ref in reference_batch:
                # Extract the tool selection part before the reasoning
                # Assumes the format: "**Tool Selection**: tools...\n\n**Reasoning**: ..."
                tool_selection_match = re.search(r'\*\*Tool Selection\*\*:\s*(.*?)\n\n\*\*Reasoning\*\*:', ref, re.DOTALL)
                if tool_selection_match:
                    tool_selection = tool_selection_match.group(1)
                else:
                    # If no reasoning is present, extract everything after '**Tool Selection**:'
                    tool_selection_match = re.search(r'\*\*Tool Selection\*\*:\s*(.*)', ref)
                    if tool_selection_match:
                        tool_selection = tool_selection_match.group(1)
                    else:
                        tool_selection = ''

                # Extract tool names using regex
                tools_in_ref = tool_pattern.findall(tool_selection)
                extracted_ref_tools = [tool for tool in tools_in_ref]
                expected_tool_sets.append(set(extracted_ref_tools))
            extracted_expected_tools.extend(expected_tool_sets)

            # --------------------------
            # Compute set-based metrics for each sample
            # --------------------------
            for pred_set, ref_set in zip(predicted_tool_sets, expected_tool_sets):
                correct = len(pred_set.intersection(ref_set))
                total_correct += correct
                total_predicted += len(pred_set)
                total_reference += len(ref_set)
                all_predicted_tool_sets.append(pred_set)
                all_reference_tool_sets.append(ref_set)

            # --------------------------
            # Debug print statements after processing
            # --------------------------
            print("Prompts batch:", prompts_batch)
            print("Reference completions batch:", reference_batch)

            # --------------------------
            # Dump the current batch's details to the JSONL file
            # --------------------------
            with open(output_jsonl, 'a') as f:
                for i in range(len(prompts_batch)):
                    f.write(json.dumps({
                        'prompt': prompts_batch[i],
                        'model_output': decoded_preds[i],
                        'extracted_output': processed_completions[i],
                        'expected_output': reference_batch[i]
                    }) + '\n')

    # --------------------------
    # Compute set-based metrics: Precision, Recall, F1 Score
    # --------------------------
    precision = (total_correct / total_predicted) * 100 if total_predicted > 0 else 0.0
    recall = (total_correct / total_reference) * 100 if total_reference > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --------------------------
    # Compute ROUGE scores on tool selections
    # --------------------------
    # Extract tool selection strings from expected completions
    reference_tool_selections = []
    for ref in reference_completions:
        # Extract the tool selection part before the reasoning
        tool_selection_match = re.search(r'\*\*Tool Selection\*\*:\s*(.*?)\n\n\*\*Reasoning\*\*:', ref, re.DOTALL)
        if tool_selection_match:
            tool_selection = tool_selection_match.group(1)
        else:
            # If no reasoning is present, extract everything after '**Tool Selection**:'
            tool_selection_match = re.search(r'\*\*Tool Selection\*\*:\s*(.*)', ref)
            if tool_selection_match:
                tool_selection = tool_selection_match.group(1)
            else:
                tool_selection = ''

        reference_tool_selections.append(tool_selection)

    # Use the processed_completions which are the extracted_output (comma-separated tools) as predictions
    predictions_tool_selections = generated_completions

    # Compute ROUGE scores on tool selections
    rouge_result = rouge.compute(predictions=predictions_tool_selections, references=reference_tool_selections, use_stemmer=True)

    # Scale the ROUGE scores to percentages
    rouge_result = {key: value * 100 for key, value in rouge_result.items()}

    # Round the ROUGE scores for readability
    rouge_result = {k: round(v, 4) for k, v in rouge_result.items()}

    # --------------------------
    # Compile all evaluation metrics
    # --------------------------
    metrics = {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'rouge1': round(rouge_result.get('rouge1', 0.0), 4),
        'rouge2': round(rouge_result.get('rouge2', 0.0), 4),
        'rougeL': round(rouge_result.get('rougeL', 0.0), 4),
        'rougeLsum': round(rouge_result.get('rougeLsum', 0.0), 4)
    }

    # --------------------------
    # Print and return the evaluation results
    # --------------------------
    print(f"Saved evaluation details to '{output_jsonl}'")
    print(f"Evaluation Metrics:")
    print(f"Precision: {metrics['precision']}%")
    print(f"Recall: {metrics['recall']}%")
    print(f"F1 Score: {metrics['f1']}%")
    print(f"ROUGE1: {metrics['rouge1']}%")
    print(f"ROUGE2: {metrics['rouge2']}%")
    print(f"ROUGE-L: {metrics['rougeL']}%")
    print(f"ROUGE-Lsum: {metrics['rougeLsum']}%")

    return metrics, generated_completions, reference_completions, prompts

# --------------------------
# Parse command-line arguments
# --------------------------
def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune and evaluate Mistral22B model with LoRA adapters.")

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
        logger.info("Only evaluation mode activated. Skipping training and interactive modes.")

        # Ensure that a saved model exists
        if not os.path.exists(save_directory):
            logger.error(f"Save directory '{save_directory}' does not exist. Cannot perform evaluation.")
            exit(1)

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Codestral-22B-v0.1",
            cache_dir=cache_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=False,
            use_cache=False,
            token=access_token
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Load the LoRA adapters
        model = PeftModel.from_pretrained(model, save_directory)
        model.eval()
        model.config.use_cache = True

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Codestral-22B-v0.1",
            cache_dir=cache_dir,
            padding_side='right'
        )

        # Set the pad_token to eos_token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --------------------------
        # Load the Evaluation Dataset
        # --------------------------
        eval_data_path = 'eval_prompt_completion_mistral.jsonl'
        eval_data = load_eval_jsonl(eval_data_path)
        eval_dataset = Dataset.from_dict(eval_data)

        # --------------------------
        # Tokenize the Evaluation Dataset
        # --------------------------
        print("Original columns in eval_dataset:", eval_dataset.column_names)

        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=[],  # Do not remove columns
            load_from_cache_file=False  # Ensure changes take effect
        )

        # Set format for PyTorch tensors, include 'prompt' and 'completion'
        tokenized_eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'labels'],
            output_all_columns=True  # Retain 'prompt' and 'completion'
        )

        print("Columns in tokenized_eval_dataset after tokenization:")
        print(tokenized_eval_dataset.column_names)

        # Optional: Print a sample to verify
        print("Sample entry from tokenized_eval_dataset:")
        print(tokenized_eval_dataset[0])

        # --------------------------
        # Perform Evaluation
        # --------------------------
        num_samples = 20
        evaluation_results, generated_completions, reference_completions, prompts = evaluate_model(
            model=model,
            tokenizer=tokenizer,
            eval_dataset=tokenized_eval_dataset,
            device='cuda',
            batch_size=1,
            max_new_tokens=40,
            num_beams=3,
            num_samples=num_samples,
            seed=42
        )
        print("\nEvaluation Results (ROUGE scores):")
        print(evaluation_results)

        exit(0)  # Exit after evaluation

    if args.interactive:
        if not os.path.exists(save_directory):
            logger.error(f"Save directory '{save_directory}' does not exist. Cannot enter interactive mode.")
            exit(1)
        logger.info(f"Loading the trained model from '{save_directory}' for interactive chat.")

        # Load the base model
        model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Codestral-22B-v0.1",
            cache_dir=cache_dir,
            device_map='auto',
            torch_dtype=torch.float16,
            quantization_config=bnb_config,    # Use quantization_config instead of load_in_8bit=True
            trust_remote_code=False,
            use_cache=False,
            token=access_token
        )

        model.config.use_cache = False
        model.config.pretraining_tp = 1
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # Load the LoRA adapters using PeftModel.from_pretrained
        model = PeftModel.from_pretrained(model, save_directory)
        model.eval()
        model.config.use_cache = True

        # Load the tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Codestral-22B-v0.1",
            cache_dir=cache_dir,
            padding_side='right'
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
            inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=4096, padding='max_length').to(model.device)
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

    if args.train:
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
        # Dump prompt and completion pairs to JSONL files
        # --------------------------
        dump_prompt_completion(train_dataset, 'train_prompt_completion_mistral.jsonl')
        dump_prompt_completion(eval_dataset, 'eval_prompt_completion_mistral.jsonl')

        # --------------------------
        # Load the tokenizer
        # --------------------------
        model_id = 'mistralai/Codestral-22B-v0.1'
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side='right',
            token = access_token
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
            remove_columns=[],  # Do not remove columns
            load_from_cache_file=False  # Ensure changes take effect
        )

        tokenized_eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=[],  # Do not remove columns
            load_from_cache_file=False  # Ensure changes take effect
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
        # Print sample entries after tokenization
        # --------------------------
        print("Sample from tokenized_train_dataset:")
        print(tokenized_train_dataset[0])

        print("Sample from tokenized_eval_dataset:")
        print(tokenized_eval_dataset[0])

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
                quantization_config=bnb_config,    # Use quantization_config instead of load_in_8bit=True
                trust_remote_code=False,
                use_cache=False,
                token=access_token
            )

            model.config.use_cache = False
            model.config.pretraining_tp = 1
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
                quantization_config=bnb_config,    # Use quantization_config instead of load_in_8bit=True
                trust_remote_code=False,
                use_cache=False,
                token=access_token
            )

            # Set use_cache to False to avoid incompatibility with gradient checkpointing
            model.config.use_cache = False
            model.config.pretraining_tp = 1

            # Enable gradient checkpointing
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

            # Prepare the model for k-bit (4-bit) training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

            # --------------------------
            # Define the target module names
            # --------------------------
            target_module_names = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

            # --------------------------
            # Apply LoRA configurations with correct target modules
            # --------------------------
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_module_names,  # Use appropriate module names
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
                per_device_train_batch_size=5,               # Keep batch size small due to potential GPU memory constraints
                per_device_eval_batch_size=5,
                dataloader_num_workers=8,                    # Number of data loader workers
                gradient_accumulation_steps=2,               # Increase to simulate a larger effective batch size
                num_train_epochs=12,                          # Adjusted number of epochs
                learning_rate=1e-5,                          # Lower learning rate for finer weight updates
                weight_decay=0.0,                            # Remove weight decay to reduce regularization
                logging_dir='./logs',                        # Directory for logging
                logging_steps=100,                           # Increase logging frequency for better monitoring
                save_strategy="steps",                       # Save checkpoints based on steps
                save_steps=500,                              # Save every 500 steps
                save_total_limit=10,                         # Limit to the last 10 checkpoints to save storage
                evaluation_strategy="steps",                 # Enable evaluation
                eval_steps=500,                              # Evaluate every 500 steps
                load_best_model_at_end=True,                 # Load the best model based on evaluation metric
                metric_for_best_model="loss",                # Monitor loss for selecting the best model
                fp16=True,                                   # Use mixed precision for faster training
                optim="adamw_hf",                            # Use AdamW optimizer
                lr_scheduler_type="linear",                  # Use a linear scheduler for simplicity
                warmup_steps=100,                            # Minimal warmup steps to stabilize training start
                max_grad_norm=1.0,                           # Enable gradient clipping
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

        if not args.interactive:
            logger.info("Starting evaluation...")

            # Perform Evaluation with Generation and Compute ROUGE Metrics
            num_samples = 20
            evaluation_results, generated_completions, reference_completions, prompts = evaluate_model(
                model=model,
                tokenizer=tokenizer,
                eval_dataset=tokenized_eval_dataset,
                device='cuda',
                batch_size=1,
                max_new_tokens=40,
                num_beams=3,
                num_samples=num_samples,
                seed=42
            )

            print("\nEvaluation Results (ROUGE scores):")
            print(evaluation_results)

if __name__ == "__main__":
    main()
