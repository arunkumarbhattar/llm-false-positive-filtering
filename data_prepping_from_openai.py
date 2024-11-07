import config
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.tools import Tool
import torch
import dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import re
from tqdm import tqdm
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    is_bug: bool = Field(
        description="Classification result. True for true positive, False for false positive"
    )
    explaination: str = Field(description="Explain your reasons for classification")

# Define tools
def get_func_definition(func_name: str) -> str:
    logger.debug(f"Fetching function definition for: {func_name}")
    res = ds.get_function_definition(func_name)
    if not res or len(res) == 0:
        return f"The definition of {func_name} is unavailable."
    return res[:10000]

def variable_def_finder(filename: str, lineno: int, varname: str) -> str:
    logger.debug(f"Finding definitions for variable '{varname}' in {filename} at line {lineno}")
    return ds.variable_def_finder(filename, lineno, varname)

def get_path_constraint(filename: str, lineno: int) -> str:
    logger.debug(f"Retrieving path constraint for {filename} at line {lineno}")
    return ds.get_path_constraint(filename, lineno)

tools = [
    Tool(
        name="get_func_definition",
        func=get_func_definition,
        description="Get the definition of a function. Use when you need to find a function's code.",
    ),
    Tool(
        name="variable_def_finder",
        func=variable_def_finder,
        description="Find where a variable is defined in the code.",
    ),
    Tool(
        name="get_path_constraint",
        func=get_path_constraint,
        description="Get the path constraint leading to a specific code line.",
    ),
]

# Define the system prompt
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are a software security researcher tasked with processing alerts generated from CodeQL, a static analysis tool.

The user will provide the alert to be processed. It contains the type of bug the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.

Please adhere to the following guidelines:

* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.

* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.

* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.

* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.
        """
         ),
        ("user", "Type of bug: {bug_type}\nLine number: {lineno}\nMessage: {msg}\nFunction code:\n{func_code}\n\nGuidance on triaging this type of bug: {guidance}"),
        ("placeholder", "{placeholder}")
    ]
)

# Define the Step 1 Prompt Template with escaped braces
# Define the Step 1 Prompt Template with all necessary fields
step1_prompt_template = PromptTemplate(
    input_variables=["bug_type", "msg", "func_code", "guidance"],
    template="""
You are a software security researcher processing a CodeQL alert.

Given the following bug information:

Type of bug: {bug_type}
Message: {msg}
Function code:
{func_code}

Guidance:

- {guidance}

Available tools:

- get_func_definition: Get the definition of a function. Use when you need to find a function's code.
- variable_def_finder: Find where a variable is defined in the code.
- get_path_constraint: Get the path constraint leading to a specific code line.

Determine which tools to invoke to assist in triaging this bug.
Provide your answer as a comma-separated list of tool names.
""".strip(),
)

model="gpt-4o-2024-05-13"
# Initialize the dataset before initializing the tokenizer and model
repo_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"

# Initialize the dataset
ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

logger.info("Starting main processing loop...")

toolstr2callable = {
    "get_func_definition": get_func_definition,
    "variable_def_finder": variable_def_finder,
    "get_path_constraint": get_path_constraint,
    "responseformatter": ResponseFormatter
}
tool_list = [
    get_func_definition,
    variable_def_finder,
    get_path_constraint,
    ResponseFormatter
]

llm = ChatOpenAI(temperature=0.2, model=model)
llm_with_tools = llm.bind_tools(tool_list)

chain = chat_prompt | llm_with_tools

# Initialize training data list
data = []

# Main loop to process each CodeQL alert
# Main loop to process each CodeQL alert
for srcfile, lineno, msg, func, gt in ds.get_codeql_results_and_gt():
    if not srcfile.endswith("_11.c"):
        logger.debug(f"Skipping file {srcfile} as it does not end with '_11.c'.")
        continue

    prompt_dict = {
        "bug_type": "Potentially uninitialized local variable",
        "filename": srcfile,  # Will exclude this in bug_info
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": """
The warning at a specific source line is a false positive if the variable is always initialized along all the paths that reach that line.
""",
    }

    # Exclude filename from bug_info
    bug_info = f"""Type of bug: {prompt_dict['bug_type']}
Line number: {prompt_dict['lineno']}
Message: {prompt_dict['msg']}
Function code:
{prompt_dict['func_code']}
"""

    guidance = prompt_dict['guidance']

    tool_names = ", ".join([tool.name for tool in tools])

    # Step 1: Determine which tools to call and their arguments
    # Step 1: Determine which tools to call and their arguments
    prompt_dict = {
        "bug_type": prompt_dict['bug_type'],
        "filename": srcfile,
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": """
The warning at a specific source line is false positive if the variable is always initialized along all the paths that reach that line.
""",
        "placeholder": []
    }


    logger.info(f"Running Step 1 LLM for file {srcfile} at line {lineno}...")
    try:
        # Invoke the LLM with the formatted prompt
        step1_response = chain.invoke(prompt_dict)
    except Exception as e:
        logger.error(f"Error running Step 1 LLM: {e}")
        continue

    logger.info("Step 1 LLM response received.")
    logger.debug(f"Step 1 Response: {step1_response}")

    # Extract tools selected as ground truth
    # Convert it to JSON string
    try:
        step1_response_json = step1_response.to_json()
    except AttributeError as e:
        logger.error(f"Error converting AIMessage to JSON: {e}")
        logger.debug(f"step1_response: {step1_response}")
        continue  # Skip to the next iteration or handle accordingly

    # Parse the JSON string into a Python dictionary
    try:
        step1_response_dict = step1_response_json.values()
    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}")
        logger.debug(f"step1_response_json: {step1_response_json}")
        continue  # Skip to the next iteration or handle accordingly

    # Convert dict_values to a list to access elements by index
    step1_response_values = list(step1_response_dict)

    # Safely access the fourth element (index 3) which contains the main response dictionary
    if len(step1_response_values) > 3:
        response_section = step1_response_values[3]
    else:
        response_section = {}
        logger.warning("Expected at least 4 elements in step1_response_dict, but got fewer.")

    # Now safely access the 'tool_calls' key
    tool_calls = response_section.get('tool_calls', [])

    if not tool_calls:
        # Fallback to 'additional_kwargs' if 'tool_calls' not found at the top level
        tool_calls = response_section.get('additional_kwargs', {}).get('tool_calls', [])

    # Extract tool names from tool_calls
    tools_selected = ", ".join([tool_call.get("name", "unknown_tool") for tool_call in tool_calls])

    # Populate the step1_prompt using the template
    step1_prompt = step1_prompt_template.format(
        bug_type="Potentially uninitialized local variable",
        filename=srcfile,
        lineno=lineno,
        msg=msg,
        func_code=func,
        guidance = "The warning at a specific source line is false positive if the variable is always initialized along all the paths that reach that line."
    )

    # Add Step 1 to training data
    data.append({
        "prompt": step1_prompt,
        "completion": tools_selected
    })

# Save the data to a JSON Lines file suitable for fine-tuning
output_file = 'fine_tuning_training_data.jsonl'
try:
    with open(output_file, 'w') as f:
        for example in data:
            json.dump(example, f)
            f.write('\n')
    logger.info(f"Training data saved to {output_file}")
except Exception as e:
    logger.error(f"Failed to save training data: {e}")
