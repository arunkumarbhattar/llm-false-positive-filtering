import config

from langchain_community.llms import HuggingFacePipeline  # Updated import to langchain-community
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import initialize_agent, AgentType, Tool
from langchain import LLMChain
from pydantic import BaseModel, Field
import torch
import dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load the tokenizer with custom cache directory
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name,
    cache_dir=config.cache_dir,
    trust_remote_code=True
)

# Load the model with optional quantization and custom cache directory
if config.use_quantization:
    if config.quantization_method == 'bitsandbytes':
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map='auto',
            torch_dtype=config.torch_dtype,
            cache_dir=config.cache_dir,
            trust_remote_code=True
        )
    elif config.quantization_method == 'gptq':
        # Uncomment and adjust if using GPTQ quantization
        # from some_gptq_library import GPTQConfig  # Replace with actual import

        # gptq_config = GPTQConfig(
        #     ...  # Add GPTQ-specific configurations
        # )

        # model = AutoModelForCausalLM.from_pretrained(
        #     config.model_name,
        #     quantization_config=gptq_config,
        #     device_map='auto',
        #     torch_dtype=config.torch_dtype,
        #     cache_dir=config.cache_dir,
        #     trust_remote_code=True
        # )
        pass
    else:
        raise ValueError(f"Unknown quantization method: {config.quantization_method}")
else:
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map='auto',
        torch_dtype=config.torch_dtype,
        cache_dir=config.cache_dir,
        trust_remote_code=True
    )

# Create a pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if config.device == 'cuda' else -1,  # HuggingFace uses device indices
    torch_dtype=config.torch_dtype,
    trust_remote_code=True,
    max_length=2048,
)

# Create LangChain LLM
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define tools
from langchain.tools import tool

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

# Define tools list
tools = [get_func_definition, variable_def_finder, get_path_constraint]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5
)

# Set up the system prompt
system_prompt = """
You are a software security researcher tasked with classifying alerts generated from CodeQL, a static analysis tool. Each alert is to be classified as either a true positive (TP) or a false positive (FP).
True Positive (TP): An alert that correctly identifies a genuine security vulnerability, code defect, or performance issue.
False Positive (FP): An alert that incorrectly identifies a problem where none exists, due to misinterpretation or overly conservative analysis.

The user will provide the alert to be classified. It contains the type of bugs the CodeQL rule intends to detect, the source line of the potential bug, the message describing the bug, followed by the code of the function containing the bug, with line numbers on the left. The language of code is C/C++.

Please adhere to the following guidelines:
* Concentrate only on the bug type and location specified by the user. Do NOT consider or report on other potential bugs or issues outside the provided scope.
* Do NOT assume or speculate about any future changes or modifications to the code. Your responses should strictly reflect the current state of the code as provided.
* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.
* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.
"""

# Initialize variables
llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0
repo_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"
ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

# Main loop
for srcfile, lineno, msg, func, gt in ds.get_codeql_results_and_gt():
    if not srcfile.endswith("_11.c"):
        continue
    prompt_dict = {
        "bug_type": "Potentially uninitialized local variable",
        "filename": srcfile,
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": """
The warning at a specific source line is false positive if the variable is always initialized along all the paths that reach that line.
"""
    }

    user_input = f"""Type of bug: {prompt_dict['bug_type']}
Source file name: {prompt_dict['filename']}
Line number: {prompt_dict['lineno']}
Message: {prompt_dict['msg']}
Function code:
{prompt_dict['func_code']}

Guidance on triaging this type of bug: {prompt_dict['guidance']}
"""

    final_input = f"{system_prompt}\n\n{user_input}"

    # Run the agent
    response = agent.run(final_input)
    print("Agent response:")
    print(response)
    # Process the response to extract is_bug and explanation
    # You may need to parse the response to get the desired fields

    # For simplicity, we'll assume the agent outputs 'True Positive' or 'False Positive' in its response
    if 'True Positive' in response or 'TP' in response:
        is_bug = True
    elif 'False Positive' in response or 'FP' in response:
        is_bug = False
    else:
        # Unable to determine, skip
        print("Unable to determine classification. Skipping...")
        continue

    explanation = response  # Or extract the explanation from the response

    llm_decision = {'is_bug': is_bug, 'explanation': explanation}

    llm_res = "LLM.BAD" if llm_decision['is_bug'] else "LLM.GOOD"
    print(f"Ground Truth: {gt}, LLM Decision: {llm_res}")
    print(f"Line Number: {lineno}\nMessage: {msg}\nFunction Code:\n{func}\nExplanation: {llm_decision['explanation']}\n----------------\n")

    if gt == dataset.GroundTruth.BAD and llm_decision['is_bug']:
        llm_tp += 1
    elif gt == dataset.GroundTruth.BAD and not llm_decision['is_bug']:
        llm_fn += 1
    elif gt == dataset.GroundTruth.GOOD and llm_decision['is_bug']:
        llm_fp += 1
    elif gt == dataset.GroundTruth.GOOD and not llm_decision['is_bug']:
        llm_tn += 1

print(f"LLM Precision: {llm_tp}/{llm_tp + llm_fp}")
print(f"LLM Recall: {llm_tp}/{llm_tp + llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
