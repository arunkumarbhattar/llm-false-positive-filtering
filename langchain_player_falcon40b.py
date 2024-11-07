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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load the tokenizer with custom cache directory
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    cache_dir=config.cache_dir,
)
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if not set

# Load the model with optional quantization and custom cache directory
logger.info("Loading model...")
if config.use_quantization:
    if config.quantization_method == 'bitsandbytes':
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map='cuda',
            torch_dtype=config.torch_dtype,
            cache_dir=config.cache_dir,
        )
    elif config.quantization_method == 'gptq':
        pass
    else:
        raise ValueError(f"Unknown quantization method: {config.quantization_method}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map='cuda',
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
    )

# Create a pipeline without specifying the 'device' parameter
logger.info("Creating text generation pipeline...")
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=config.torch_dtype,
    trust_remote_code=False,
    max_new_tokens=200,
    truncation=True,
    padding='max_length',
)

# Create LangChain LLM
logger.info("Creating LangChain LLM...")
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define tools
from langchain.tools import StructuredTool

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

# Define tools using StructuredTool
get_func_definition_tool = StructuredTool.from_function(
    func=get_func_definition,
    name="get_func_definition",
    description="Get the definition of a function. Use when you need to find a function's code.",
)

variable_def_finder_tool = StructuredTool.from_function(
    func=variable_def_finder,
    name="variable_def_finder",
    description=(
        "Finds all definitions of a local variable in a specified function within a given file. "
        "Use when you need to find where a variable is defined."
    ),
)

get_path_constraint_tool = StructuredTool.from_function(
    func=get_path_constraint,
    name="get_path_constraint",
    description=(
        "Retrieves the path constraint for a specific source code location. "
        "Use when you need to understand the conditions leading to a code location."
    ),
)

tools = [get_func_definition_tool, variable_def_finder_tool, get_path_constraint_tool]

# Prepare the system prompt
system_prompt = """
You are a software security researcher tasked with classifying alerts generated from CodeQL, a static analysis tool.

True Positive (TP): An alert that correctly identifies a genuine security vulnerability, code defect, or performance issue.

False Positive (FP): An alert that incorrectly identifies a problem where none exists, due to misinterpretation or overly conservative analysis.

Please adhere to the following guidelines:

* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.

* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.

* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.

* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.

The user will provide the alert to be classified. It contains the type of bugs the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.
"""

# Define the format instructions
format_instructions = """
Use the following format:

Question: the input question you must answer

Thought: your reasoning process

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now have enough information

Answer: the final answer to the original question

Remember to use the tools when necessary to find additional information.
"""

# Prepare the tool names
tool_names = ", ".join([tool.name for tool in tools])

# Create the prompt template
prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template=f"""
{system_prompt}

{format_instructions}

Question: {{input}}

{{agent_scratchpad}}
""".replace("{tool_names}", tool_names),
)

# Initialize the LLMChain with the custom prompt
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

# Create the ZeroShotAgent
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    tools=tools,
    verbose=True,
)

# Create the AgentExecutor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
)

# Initialize variables
llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0
repo_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3/"
sarif_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/scratch/gilbreth/bhattar1/llm-false-positive-filtering/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"
ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

logger.info("Starting main processing loop...")
# Main loop
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

    final_input = user_input  # Do not include the system prompt here

    logger.info(f"Running agent for file {srcfile} at line {lineno}...")
    try:
        response = agent_executor.run(final_input)
    except Exception as e:
        logger.error(f"Error running the agent: {e}")
        continue

    logger.info("Agent response received.")
    logger.debug(f"Response: {response}")

    # Parse the response
    is_bug = None
    explanation = response

    # Define regex patterns for parsing
    tp_pattern = re.compile(r'\b(True Positive|TP)\b', re.IGNORECASE)
    fp_pattern = re.compile(r'\b(False Positive|FP)\b', re.IGNORECASE)

    if tp_pattern.search(response):
        is_bug = True
        logger.debug("Parsed response as True Positive.")
    elif fp_pattern.search(response):
        is_bug = False
        logger.debug("Parsed response as False Positive.")
    else:
        # Unable to determine, skip
        logger.warning("Unable to determine classification from the response. Skipping...")
        continue

    llm_decision = {'is_bug': is_bug, 'explanation': explanation}

    llm_res = "LLM.BAD" if llm_decision['is_bug'] else "LLM.GOOD"
    logger.info(f"Ground Truth: {gt}, LLM Decision: {llm_res}")
    logger.debug(
        f"Line Number: {lineno}\nMessage: {msg}\nFunction Code:\n{func}\nExplanation: {llm_decision['explanation']}\n----------------\n"
    )

    # Update counters based on ground truth and model decision
    if gt == dataset.GroundTruth.BAD and llm_decision['is_bug']:
        llm_tp += 1
    elif gt == dataset.GroundTruth.BAD and not llm_decision['is_bug']:
        llm_fn += 1
    elif gt == dataset.GroundTruth.GOOD and llm_decision['is_bug']:
        llm_fp += 1
    elif gt == dataset.GroundTruth.GOOD and not llm_decision['is_bug']:
        llm_tn += 1

# Final evaluation metrics
logger.info("Processing complete. Calculating evaluation metrics...")
print(f"LLM Precision: {llm_tp}/{llm_tp + llm_fp}")
print(f"LLM Recall: {llm_tp}/{llm_tp + llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
