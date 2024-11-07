from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType, initialize_agent, Tool
import json
import sys

from repository_manager import RepositoryManager, GroundTruth
from tools import tool_list  # Ensure tools.py defines 'tool_list' as a list of Tool objects

# Define the chat prompt
chat_prompt = ChatPromptTemplate.from_messages(
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
    ]
)

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""
    is_bug: bool = Field(
        description="Classification result. True for true positive, False for false positive"
    )
    explanation: str = Field(description="Explain your reasons for classification")

# Keep the model as per your original code
model = "gpt-4o-mini-2024-07-18"

# Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.2, model=model)

# Initialize the Agent with tools
agent = initialize_agent(
    tools=tool_list,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

# Initialize the RepositoryManager
repo_path = sys.argv[1]
sarif_path = sys.argv[2]
bitcode_path = sys.argv[3]
repo = RepositoryManager(repo_path, sarif_path, bitcode_path)

# Load guidance data
with open("guidance.json", 'r') as json_file:
    guidance = json.load(json_file)

# Initialize counters for evaluation metrics
llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0

# Iterate over CodeQL results and ground truths
for srcfile, lineno, msg, func, gt, rule_id, rule_desc in repo.get_codeql_results_and_gt():
    if func is None:
        print(f"Skipping analysis for file: {srcfile}, line: {lineno} due to missing function location.")
        continue

    prompt_dict = {
        "bug_type": rule_desc,
        "filename": srcfile,
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": guidance.get(rule_id, ""),
    }

    # Construct the full prompt
    full_prompt = chat_prompt.format_messages(**prompt_dict)

    # Run the agent with the prompt
    try:
        response = agent(full_prompt)
    except Exception as e:
        print(f"Error during agent run: {e}")
        continue

    # Assuming the agent returns a JSON string that matches ResponseFormatter
    try:
        formatted_response = ResponseFormatter.parse_raw(response['output'])
    except Exception as e:
        print(f"Error parsing response: {e}")
        continue

    llm_res = "LLM.BAD" if formatted_response.is_bug else "LLM.GOOD"
    print(f"Ground Truth: {gt}, LLM Decision: {llm_res}")
    print('===========CASE END=================')

    # Update evaluation metrics
    if gt == GroundTruth.BAD and formatted_response.is_bug:
        llm_tp += 1
    elif gt == GroundTruth.BAD and not formatted_response.is_bug:
        llm_fn += 1
    elif gt == GroundTruth.GOOD and formatted_response.is_bug:
        llm_fp += 1
    elif gt == GroundTruth.GOOD and not formatted_response.is_bug:
        llm_tn += 1

# Print evaluation metrics
print(f"LLM precision: {llm_tp}/{llm_tp+llm_fp}")
print(f"LLM recall: {llm_tp}/{llm_tp+llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
