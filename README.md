# CodeQL False Positive Filtering Using LLMs and LLVM

## Overview:
This is to allow to increase the accuracy of CodeQL static analysis by using LLMs and LLVM backend tools. Specifically, we introduce several LLVM passes as "fact-finders". We in-detail expose these LLVM passes as tools with detailed explanation as to how to use that tool and what does this tool exactly analyze. The LLM, given the description of the bug will be able to decide which set of facts it needs to call to retrieve information on whether this is a false positive or true positive result from codeQL.

#Directory Structure

The project is organized into several directories, each containing scripts and tools tailored for specific tasks within the false positive filtering pipeline.
```aiignore

llm-false-positive-filtering/
├── finetuning_custom_llms/
│   ├── config.py
│   ├── finetuning_codellama.py
│   ├── finetuning_codellama_with_reasoning.py
│   └── finetuning.py
├── ground_truth_generation/
│   ├── add_ground_truth.py
│   ├── dataset_juliet.py
│   ├── filter.sh
│   ├── parse.py
│   └── README.md
├── llvm/
│   ├── build/
│   ├── CMakeLists.txt
│   ├── include/
│   ├── lib/
│   ├── libs/
│   └── README.md
├── prompt_pair_prepping/
│   ├── data_prepping_all_facts_from_openai.py
│   ├── dataset.py
│   ├── fine_tuning_training_data.jsonl
│   ├── guidance.json
│   ├── langchain-player.py
│   ├── llm_triage.py
│   ├── prompts.py
│   ├── repository_manager.py
│   └── tools.py
├── __pycache__/
├── prepare-codeql-tests.sh
├── README.md
├── requirements_for_ground_truth_generation.txt
└── venv/

```

## directory --> finetuning_custom_llms

This directory contain the set of scripts that are used to fine-tune LLMs using prompt-action pairs.
Specifically, we train the LLM to be able to, given information on the bug detected by CodeQL, identify which subset of tools it needs to call from its arsenal to verify the authenticity of this bug.
At this moment, the custom fine-tuned LLMs do not invoke the tool itself, but they just identify the tools that need to be used. 
Incorporating the logic to automatically call the tools and verify output from tools is only done by OpenAI GPT-4o and not these custom finetuned LLMs at the moment.
However, in future we will do it!


### finetuning scripts -->

The scripts differ in model trained on, and data used for training. 
In general the scripts employ the following pipeling -->

#### Script Modes 
    ##### Training mode: Fine-tune the model using Qlora and Lora adapters
    ##### Evaluation mode: Assess the model's performance using ROUGE and F1-score metrics
    ##### Interactive mode: Give user the chat interface 
    ##### Re-training mode: Retrain the full model from scratch

## directory --> prompt_pair_prepping

### data_prepping_all_facts_from_openai.py
#### Purpose: 
Automate the process of gather and preprocessing factual data from OpenAI sources. This data will eventually be used in training LLMs.

#### Relevance: 
Automates the creation of high quality prompt-completion pairs by using ChatGPT to interpret and process CodeQL alerts. This ensures that the fine-tuning ground truth data accurately reflects the types of analyses the model with perform.

#### Initialization:
    Set up logging and initialize GPT-4o with specific temp settings.
    Load guidance
#### Agent initialization:
    Initialize an agent with set of predefined tools and system messages.
#### Loopin em':
    Iterate through CodeQL alerts retrieved from repo
    For each alert, construct a user message containing bug detailed and guidance (retrieved from guidance.json)
    Runs the agent to generate responses.
    Extract all relevant data from AI message output
    Format the extracted data into prompt-answer pairs and save it into fine_tuning_training_data.json file

### dataset.py
#### Purpose:
    Manages the dataset used for training and evaluating the LLM's ability to filter false positives in CodeQL analysis.

#### Relevance:
    Serves as the backbone for data management within the project, enabling the extraction, labeling, and preparation of CodeQL findings for training and evaluation.

#### Parameters:
Initialization Parameters:

    repo_path: Path to the code repository being analyzed.
    sarif_path: Path to the CodeQL SARIF report file.
    bitcode_path: Path to the LLVM bitcode file.

#### Functionality: 
Functionality:

    Initialization:
        Sets up paths to repository, SARIF reports, and bitcode files.
        Generates and loads ctags to map function identifiers to their definitions.

    Data Retrieval:
        get_codeql_results_and_gt: Parses SARIF reports to extract CodeQL findings along with their ground truth labels.
        get_codeql_results_and_gt_all_facts: Similar to the above but includes additional metadata like ruleId.

    Manual Labeling:
        manually_label_codeql_results: Allows users to manually assign ground truth labels (TP/FP) to CodeQL results, facilitating the creation of accurate training data.

    Helper Methods:
        Functions to retrieve function definitions, dump source code snippets, and interact with LLVM tools for deeper code analysis.

### llm_triage.py
#### Purpose:

    Basically godfather of all scripts. Use chatgpt LLM to make decide which tools to use, and based on tool, invoke that LLM took and make the bug prediction as TP or FP.


Acts as the central script for adapting the LLM to the specific task of triaging CodeQL alerts. By managing both the training and interactive aspects, it ensures that the model can be fine-tuned effectively and utilized seamlessly within different operational modes, enhancing the project's ability to filter false positives accurately.

## Step 1: Compile LLVM Passes
Refer to `llvm/README.md` for detailed instructions on how to compile the necessary LLVM passes.
Then set the environment variable
``` bash
export LLVM_PASSES_LIB_DIR=/path/to/llm-sast-triage/llvm/build/lib
```

## Step 2: Run the LLM
To analyze warnings in a repository, provide the following inputs:
- Path to the repository
- CodeQL SARIF file
- LLVM bitcode file

### Usage:
```bash
python3 llm_triage.py <repo_path> <sarif_path> <bitcode_path>
```
