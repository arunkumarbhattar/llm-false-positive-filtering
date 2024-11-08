import config
import torch
import logging
import re
import json
from tqdm import tqdm
from typing import List, Any

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain.schema import AgentAction, AgentFinish
from langchain.pydantic_v1 import BaseModel, Field

import dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Initialize the dataset
repo_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"

ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

# Define tools using the @tool decorator
@tool
def get_func_definition(func_name: str) -> str:
    """Get the definition of a function.
    Use the tool when you want to lookup the definition of a function
    whose function body has not been provided to you,
    AND you think the definition of this function is crucial to your decision.
    """

    res = ds.get_function_definition(func_name)
    if not res or len(res) == 0:
        return f"The definition of {func_name} is unavailable."
    return res[:10000]

@tool
def variable_def_finder(filename: str, lineno: int, varname: str) -> str:
    """
    Finds all definitions of a local variable in a specified function within a given file.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file to analyze.
    lineno : int
        The line number in the source file where the function containing the variable is defined.
    varname : str
        The name of the local variable whose definitions are to be found.

    Returns:
    --------
    str
        A string containing the details of the line numbers of all definitions of the specified
        variable. If no definitions are found, the function may return an empty string or an error
        message depending on the analysis result.
    """
    return ds.variable_def_finder(filename, lineno, varname)

@tool
def get_path_constraint(filename: str, lineno: int) -> str:
    """
    Retrieves the path constraint for a specific source code location.

    This function analyzes the code at the given file and line number, extracting the
    logical path constraint that leads to the specified location. Path constraints
    are the conditions that must be true for the execution to reach the given line
    in the code.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file to analyze.
    lineno : int
        The line number in the source file for which the path constraint is to be determined.

    Returns:
    --------
    str
        A string representation of the path constraint that leads to the specified line.
        If no constraint is found or the analysis fails, an empty string or a relevant
        error message may be returned.
    """
    return ds.get_path_constraint(filename, lineno)

# Create the list of tools (but we won't execute them)
tools = [
    get_func_definition,
    variable_def_finder,
    get_path_constraint
]

# Initialize the LLM
model = "gpt-4o-2024-05-13"  # Replace with your actual model name
llm = ChatOpenAI(temperature=0.2, model=model)

# Define the agent's system message (prompt)
system_message = """
You are a software security researcher tasked with processing alerts generated from CodeQL, a static analysis tool.

The user will provide the alert to be processed. It contains the type of bug the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.

Please adhere to the following guidelines:
* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.
* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.
* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.
* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.

**Instructions:**

1. **Tool Selection**: Plan which tools to invoke to assist in triaging this bug. Do not execute any tool. You might need 1 or more tools at the same time. Provide your answer in a comma seperated format

2. **Reasoning**: Provide the reasoning behind your selection in plain English, using a chain-of-thought format.

""".strip()

agent_kwargs = {
    "system_message": system_message
}

# Initialize the agent with the tools and system message, but prevent tool execution
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    agent_kwargs=agent_kwargs,
    agent_executor_kwargs={
        "return_intermediate_steps": True,
        "max_iterations": 1  # Ensure the agent does not attempt to execute tools
    },
    verbose=True,
)

logger.info("Starting main processing loop...")

# Initialize training data list
data = []

# Main loop to process each CodeQL alert
for srcfile, lineno, msg, func, gt in ds.get_codeql_results_and_gt():
    if not srcfile.endswith("_11.c"):
        logger.debug(f"Skipping file {srcfile} as it does not end with '_11.c'.")
        continue

    prompt_dict = {
        "bug_type": "Potentially uninitialized local variable",
        "filename": srcfile,
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": """
The warning at a specific source line is a false positive if the variable is always initialized along all the paths that reach that line.
""",
    }

    logger.info(f"Processing file {srcfile} at line {lineno}...")

    try:
        # Prepare the user message
        user_message = f"""
Type of bug: {prompt_dict['bug_type']}
Line number: {prompt_dict['lineno']}
Message: {prompt_dict['msg']}
Function code:
{prompt_dict['func_code']}

Guidance on triaging this type of bug: {prompt_dict['guidance'].strip()}
"""
        # Run the agent (without executing tools)
        result = agent.run(user_message)
    except Exception as e:
        logger.error(f"Error running agent: {e}")
        continue

    response = result  # The agent's response

    # First, try to match "Tools to invoke: tool1, tool2"
    tools_selected_match = re.search(r"Tools to invoke:\s*([^\n]*)", response, re.IGNORECASE)
    if tools_selected_match:
        # Split the tools by comma and strip any surrounding whitespace
        tools_selected_str = ", ".join([tool.strip() for tool in tools_selected_match.group(1).split(",")])
        # Remove the "Tools to invoke: ..." line from reasoning
        reasoning = response.replace(tools_selected_match.group(0), "").strip()
    else:
        tools_selected_str = ""
        reasoning = response.strip()

    # Optionally, ensure that reasoning starts after "Reasoning:"
    reasoning_match = re.search(r"Reasoning:\s*(.*)", reasoning, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Prepare the instruction, input, and output for training data
    instruction = "Analyze the provided CodeQL alert and determine which tools to invoke to assist in triaging the bug. Provide the tools and your reasoning."
    input_text = user_message.strip()
    output_text = response.strip()

    # Add to training data
    data.append({
        "instruction": instruction,
        "input": input_text,
        "output": output_text
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
