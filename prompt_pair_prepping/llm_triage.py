import os
import json
import sys
import argparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from repository_manager import RepositoryManager, GroundTruth
from tools import *
import tools
# ---------------------------- Chat Prompt Template ----------------------------

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a software security researcher tasked with classifying alerts generated from CodeQL, a static analysis tool. Each alert is to be classified as either a true positive (TP) or a false positive (FP).
True Positive (TP): An alert that correctly identifies a genuine security vulnerability, code defect, or performance issue.
False Positive (FP): An alert that incorrectly identifies a problem where none exists, due to misinterpretation or overly conservative analysis.

The user will provide the alert to be classified. It contains the type of bugs the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.

Please adhere to the following guidelines:
* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.
* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.
* Let's think step by step. Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.
* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.
"""
         ),
        ("user", "Type of bug: {bug_type}\nSource file name: {filename}\nLine number: {lineno}\nMessage: {msg}\nFunction code:\n{func_code}\n\nGuidance on triaging this type of bug: {guidance}"),
        ("placeholder", "{history}")
    ]
)

# ---------------------------- Response Formatter ----------------------------

class ResponseFormatter(BaseModel):
    """Structure for the AI's classification response."""
    is_bug: bool = Field(
        description="Classification result. True for true positive, False for false positive"
    )
    explanation: str = Field(description="Explanation for the classification decision")

# ---------------------------- Language Model Initialization ----------------------------

# Choose the appropriate model

#MODEL_NAME="gpt-3.5-turbo-0125"
MODEL_NAME="gpt-4o-mini-2024-07-18"
#MODEL_NAME="gpt-4o-2024-05-13"

# Initialize the ChatOpenAI model with desired parameters
llm = ChatOpenAI(temperature=0.2, model=MODEL_NAME)

# Bind tools to the language model
llm_with_tools = llm.bind_tools(gpt_tool_list)

# Set up structured output using the ResponseFormatter
structured_llm = llm.with_structured_output(ResponseFormatter)

# Create processing chains
classification_chain = CHAT_PROMPT | llm_with_tools
structured_decision_chain = CHAT_PROMPT | structured_llm

# ---------------------------- Processing Function ----------------------------

def process_juliet_repo(juliet_repo_path):
    """
    Process the Juliet test suite repository for CodeQL alerts classification.

    Args:
        juliet_repo_path (str): The base path to the Juliet test suite repository.
    """
    print(f"Starting processing of Juliet repository at: {juliet_repo_path}\n")

    # Construct essential paths
    repo_path = juliet_repo_path
    guidance_file = "guidance.json"

    # Define the SARIF subpaths to process
    sarif_subpaths = [
        "C/testcases/CWE457_Use_of_Uninitialized_Variable/s02/",
        "C/testcases/CWE121_Stack_Based_Buffer_Overflow/s01/"
    ]

    # Initialize evaluation metrics
    metrics = {
        "TP": 0,  # True Positives
        "FP": 0,  # False Positives
        "FN": 0,  # False Negatives
        "TN": 0   # True Negatives
    }

    with open(guidance_file, 'r') as json_file:
        guidance = json.load(json_file)
    print("Guidance data successfully loaded.\n")

    # Iterate over each SARIF subpath
    for subpath in sarif_subpaths:
        sarif_full_path = os.path.join(juliet_repo_path, subpath, "cpp-labeled.sarif")
        print(f"Processing SARIF file: {sarif_full_path}")

        if not os.path.isfile(sarif_full_path):
            print(f"Warning: SARIF file not found at {sarif_full_path}. Skipping this subpath.\n")
            continue

        # Initialize the RepositoryManager for the current SARIF path
        bitcode_path = os.path.join(juliet_repo_path, subpath, "partial.o.bc")
        tools.repo = RepositoryManager(repo_path, sarif_full_path, bitcode_path)
        print(f"Initialized RepositoryManager for {sarif_full_path}.\n")

        # Iterate over CodeQL results and ground truth data
        for idx, (srcfile, lineno, msg, func, gt, rule_id, rule_desc) in enumerate(tools.repo.get_codeql_results_and_gt(), start=1):
            print(f"Processing alert {idx}: {srcfile}:{lineno} - {msg}")

            # Skip if guidance for the rule_id is not available
            if not guidance.get(rule_id, ''):
                print(f"No guidance found for rule ID: {rule_id}. Skipping this alert.\n")
                continue

            # Prepare the prompt dictionary
            prompt_dict = {
                "bug_type": rule_desc,
                "filename": srcfile,
                "lineno": lineno,
                "msg": msg,
                "func_code": func,
                "guidance": guidance[rule_id],
                "history": []
            }

            # Engage in the classification conversation with the AI
            while True:
                ai_response = classification_chain.invoke(prompt_dict)
                prompt_dict["history"].append(ai_response)
                print(f"AI Response: {ai_response}")

                if not ai_response.tool_calls:
                    print("No further tool calls required by AI.\n")
                    break

                # Handle each tool call requested by the AI
                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"].lower()
                    print(f"Tool Call Detected: {json.dumps(tool_call, indent=2)}")

                    selected_tool = gpt_toolstr2callable.get(tool_name)
                    if not selected_tool:
                        print(f"Error: No tool found for '{tool_name}'. Skipping this tool call.\n")
                        continue

                    tool_response = selected_tool.invoke(tool_call)
                    print(f"Tool Response: {tool_response}\n")
                    prompt_dict["history"].append(tool_response)

            # Obtain structured decision from the AI
            decision = structured_decision_chain.invoke(prompt_dict)
            decision_label = "TP" if decision.is_bug else "FP" if gt == GroundTruth.GOOD else "TN"
            print(f"Ground Truth: {gt.name}, LLM Decision: {'True Positive' if decision.is_bug else 'False Positive'}")
            print('=========== CASE END ================\n')

            # Update evaluation metrics based on ground truth and AI decision
            if gt == GroundTruth.BAD:
                if decision.is_bug:
                    metrics["TP"] += 1
                else:
                    metrics["FN"] += 1
            elif gt == GroundTruth.GOOD:
                if decision.is_bug:
                    metrics["FP"] += 1
                else:
                    metrics["TN"] += 1

    # Calculate precision and recall
    precision = (metrics["TP"] / (metrics["TP"] + metrics["FP"])) if (metrics["TP"] + metrics["FP"]) > 0 else 0
    recall = (metrics["TP"] / (metrics["TP"] + metrics["FN"])) if (metrics["TP"] + metrics["FN"]) > 0 else 0

    # Display evaluation metrics
    print("========= Evaluation Metrics =========")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print(f"True Negatives (TN): {metrics['TN']}\n")
    print(f"LLM Precision: {metrics['TP']}/{metrics['TP'] + metrics['FP']} ({precision:.2f})")
    print(f"LLM Recall: {metrics['TP']}/{metrics['TP'] + metrics['FN']} ({recall:.2f})")
    print("=======================================\n")

# ---------------------------- Main Execution ----------------------------

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process Juliet test suite for CodeQL alerts classification."
    )
    parser.add_argument(
        "--juliet_repo_path",
        type=str,
        required=True,
        help="Path to the Juliet test suite repository."
    )
    args = parser.parse_args()

    juliet_repo_path = args.juliet_repo_path

    # Validate the provided repository path
    if not os.path.isdir(juliet_repo_path):
        print(f"Error: The provided Juliet repository path does not exist or is not a directory: {juliet_repo_path}")
        sys.exit(1)

    # Start processing
    process_juliet_repo(juliet_repo_path)

if __name__ == "__main__":
    main()
