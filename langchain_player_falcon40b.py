import config

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
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

# Define prompt templates for each step
step1_prompt_template = PromptTemplate(
    input_variables=["bug_info", "guidance", "tool_names"],
    template="""
You are a software security researcher tasked with processing alerts generated from CodeQL, a static analysis tool.

Given the following bug information:

{bug_info}

And the following guidance:

{guidance}

Your task is to determine which tools to call from the available tools: {tool_names}.

List the tools you need to call, without providing any arguments.

Provide your answer as a comma-separated list of tool names.
""".strip(),
)

step2_prompt_template = PromptTemplate(
    input_variables=["tool_name", "bug_info", "guidance"],
    template="""
You have decided to call the tool "{tool_name}" based on the following bug information:

{bug_info}

And the following guidance:

{guidance}

Now, extract the necessary arguments for the tool "{tool_name}".

Provide the arguments in the following format:

Args:
  arg1: value1
  arg2: value2

Replace arg1, arg2, etc., and value1, value2, etc., with the actual argument names and their corresponding values.
""".strip(),
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

    bug_info = f"""Type of bug: {prompt_dict['bug_type']}
Source file name: {prompt_dict['filename']}
Line number: {prompt_dict['lineno']}
Message: {prompt_dict['msg']}
Function code:
{prompt_dict['func_code']}
"""

    guidance = prompt_dict['guidance']

    tool_names = ", ".join([tool.name for tool in tools])

    # Step 1: Decide which tools to call
    step1_llm_chain = LLMChain(llm=llm, prompt=step1_prompt_template)

    step1_input = {
        "bug_info": bug_info,
        "guidance": guidance,
        "tool_names": tool_names,
    }

    logger.info(f"Running step 1 LLM chain for file {srcfile} at line {lineno}...")
    try:
        step1_response = step1_llm_chain.run(step1_input)
    except Exception as e:
        logger.error(f"Error running the step 1 LLM chain: {e}")
        continue

    logger.info("Step 1 LLM response received.")
    logger.debug(f"Step 1 Response: {step1_response}")

    # Parse the response to get the list of tools
    # Assuming the response is a comma-separated list of tool names
    tool_list = [tool_name.strip() for tool_name in step1_response.split(",")]

    # Remove any empty strings
    tool_list = [tool_name for tool_name in tool_list if tool_name]

    # For each tool, get the arguments
    observations = {}
    for tool_name in tool_list:
        # Ensure the tool name is valid
        tool = next((t for t in tools if t.name == tool_name), None)
        if tool is None:
            logger.warning(f"Tool {tool_name} not found among defined tools.")
            continue

        # Step 2: Get the arguments for the tool
        step2_llm_chain = LLMChain(llm=llm, prompt=step2_prompt_template)

        step2_input = {
            "tool_name": tool_name,
            "bug_info": bug_info,
            "guidance": guidance,
        }

        logger.info(f"Running step 2 LLM chain for tool {tool_name}...")
        try:
            step2_response = step2_llm_chain.run(step2_input)
        except Exception as e:
            logger.error(f"Error running the step 2 LLM chain for tool {tool_name}: {e}")
            continue

        logger.info(f"Step 2 LLM response received for tool {tool_name}.")
        logger.debug(f"Step 2 Response for tool {tool_name}: {step2_response}")

        # Parse the arguments from the response
        # Assuming the response is in the format:
        # Args:
        #   arg1: value1
        #   arg2: value2

        args_pattern = re.compile(r'Args:\s*((?:\s+\w+:\s+.*\n?)+)', re.MULTILINE)
        args_match = args_pattern.search(step2_response)

        if not args_match:
            logger.error(f"Could not parse arguments for tool {tool_name} from response.")
            continue

        args_block = args_match.group(1)
        args = {}
        for arg_line in args_block.strip().split('\n'):
            arg_match = re.match(r'\s*(\w+):\s+(.*)', arg_line)
            if arg_match:
                arg_key = arg_match.group(1)
                arg_value = arg_match.group(2)
                args[arg_key] = arg_value
            else:
                logger.warning(f"Could not parse argument line: {arg_line}")

        # Now execute the tool function
        logger.info(f"Executing tool {tool_name} with arguments: {args}")
        try:
            observation = tool.func(**args)
            observations[tool_name] = observation
            logger.debug(f"Executed tool {tool_name}: {observation}")
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")

    # Now, we can proceed with further processing using the observations
    # For the purpose of this code, we'll log the observations
    logger.info(f"Observations for file {srcfile} at line {lineno}: {observations}")

    # Update counters based on ground truth if necessary
    # (This part remains unchanged as per your original setup)
    # ...

# Final evaluation metrics
logger.info("Processing complete. Calculating evaluation metrics...")
print(f"LLM Precision: {llm_tp}/{llm_tp + llm_fp if llm_tp + llm_fp > 0 else 1}")
print(f"LLM Recall: {llm_tp}/{llm_tp + llm_fn if llm_tp + llm_fn > 0 else 1}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
