import config

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import LLMChain
import torch
import dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import logging
import re
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Load the tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    cache_dir=config.cache_dir,
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if not set

# Load the model
logger.info("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    device_map='auto',  # Use 'auto' to automatically select the device
    torch_dtype=config.torch_dtype,
    cache_dir=config.cache_dir,
)

# Create a text generation pipeline
logger.info("Creating text generation pipeline...")
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=config.torch_dtype,
    max_new_tokens=512,  # Increase max_new_tokens to allow longer outputs
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id
)

# Create LangChain LLM
logger.info("Creating LangChain LLM...")
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define tools
from langchain.tools import Tool

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

# Prepare the system prompt
system_prompt = """
You are a software security researcher tasked with processing alerts generated from CodeQL, a static analysis tool.

Your sole responsibility is to determine which tools to invoke based on the type of bug, its location, and the provided guidance. You must extract the necessary parameters for each tool call without making any classification decisions.

Please adhere to the following guidelines:

* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.

* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.

* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.

* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.

The user will provide the alert to be processed. It contains the type of bug the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.

Your task is to determine which tools to call based on the guidance and the bug alert information. For each tool you decide to call, specify the tool name and provide the necessary parameters.
"""

# Define the format instructions
format_instructions = """
Use the following format:

Tool Calls:
  [tool_name_1] ([call_id_1])
 Call ID: [call_id_1]
  Args:
    [arg1]: [value1]
    [arg2]: [value2]
  [tool_name_2] ([call_id_2])
 Call ID: [call_id_2]
  Args:
    [arg1]: [value1]
    [arg2]: [value2]

Replace [tool_name_n], [call_id_n], [argn], and [valuen] with the actual tool names, unique call identifiers, argument names, and their corresponding values.
"""

# Prepare the tool names
tool_names = ", ".join([tool.name for tool in tools])

# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=f"""
{system_prompt}

{format_instructions}

Tool Calls:
{{agent_scratchpad}}
""".strip(),
)

# Initialize the LLMChain with the custom prompt
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Create the ZeroShotAgent
agent = ZeroShotAgent.from_llm_and_tools(
    llm=llm,
    tools=tools,
    prefix=system_prompt,
    format_instructions=format_instructions.replace("{tool_names}", tool_names),
    max_iterations=5,
    verbose=True,
)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
)

# Initialize variables for evaluation metrics
llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0

# Define paths (ensure these paths are correct and accessible)
repo_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"

# Initialize the dataset
ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

logger.info("Starting main processing loop...")
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

    user_input = f"""Type of bug: {prompt_dict['bug_type']}
Source file name: {prompt_dict['filename']}
Line number: {prompt_dict['lineno']}
Message: {prompt_dict['msg']}
Function code:
{prompt_dict['func_code']}

Guidance on triaging this type of bug: {prompt_dict['guidance']}
"""

    final_input = user_input

    logger.info(f"Running agent for file {srcfile} at line {lineno}...")
    try:
        response = agent_executor.run(final_input)
    except Exception as e:
        logger.error(f"Error running the agent: {e}")
        continue

    logger.info("Agent response received.")
    logger.debug(f"Response: {response}")

    # Parse the response to extract tool calls
    tool_call_pattern = re.compile(
        r'(?P<tool_name>\w+)\s+\((?P<call_id>call_[\w\d]+)\)\s+Call ID:\s+(?P=call_id)\s+Args:\s+((?:\s+[\w]+:\s+.+\n)+)'
    )

    matches = tool_call_pattern.finditer(response)
    tool_calls = []

    for match in matches:
        tool_name = match.group("tool_name")
        call_id = match.group("call_id")
        args_block = match.group(4).strip()
        # Extract arguments
        args = {}
        for arg_line in args_block.split('\n'):
            arg_match = re.match(r'\s+([\w]+):\s+(.+)', arg_line)
            if arg_match:
                arg_key = arg_match.group(1)
                arg_value = arg_match.group(2)
                args[arg_key] = arg_value
        tool_calls.append({
            "name": tool_name,
            "id": call_id,
            "args": args
        })

    # Execute the tool calls and gather observations
    observations = {}
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        call_id = tool_call["id"]
        args = tool_call["args"]

        # Find the corresponding tool
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            logger.warning(f"Tool {tool_name} not found among defined tools.")
            continue

        # Execute the tool function with extracted arguments
        observation = tool.func(**args)
        observations[call_id] = observation
        logger.debug(f"Executed tool {tool_name} with Call ID {call_id}: {observation}")

    # At this point, you can use the observations to make further decisions or classifications
    # For the purpose of this modification, we'll stop here

    # Example: Logging the observations
    logger.info(f"Observations for file {srcfile} at line {lineno}: {observations}")

    # Update counters based on ground truth if necessary
    # (This part remains unchanged as per your original setup)
    # ...

# Final evaluation metrics
logger.info("Processing complete. Calculating evaluation metrics...")
print(f"LLM Precision: {llm_tp}/{llm_tp + llm_fp}")
print(f"LLM Recall: {llm_tp}/{llm_tp + llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
