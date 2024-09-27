from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

import dataset

#import langchain
#langchain.debug = True
#langchain.verbose = True

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
* Do NOT hurry to make a decision when uncertain. Use the provided tools to seek clarification or obtain necessary information.
* In C and C++, defining a variable with the static keyword inside a function only affects that specific variable. The use of static ensures that this variable retains its value between function calls and is not reinitialized on subsequent calls. However, this behavior does not apply to other variables in the same function unless they are also explicitly defined with static. Each variable's storage class and lifetime are independent of others in the function, and only variables defined with static retain their values across multiple calls.
"""
        ),
        ("user", "Type of bug: {bug_type}\nSource file name:{filename}\nLine number: {lineno}\nMessage: {msg}\nFunction code:\n{func_code}\n\nGuidance on triaging this type of bug: {guidance}"),
        ("placeholder", "{placeholder}")
    ]
)

#@tool
class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    is_bug: bool = Field(
        description="Classification result. True for true positive, False for false positive"
    )
    explaination: str = Field(description="Explain your reasons for classification")

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

# LLM
#model="gpt-3.5-turbo-0125"
model="gpt-4o-mini-2024-07-18"
#model="gpt-4o-2024-05-13"
#llm = ChatOpenAI(temperature=0.2, model=model).with_structured_output(
#    Classification
#)
llm = ChatOpenAI(temperature=0.2, model=model)
llm_with_tools = llm.bind_tools(tool_list)

chain = chat_prompt | llm_with_tools

llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0
repo_path = "/home/mjshen/IVOS/repos/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/mjshen/IVOS/repos/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-202408221915.sarif"
bitcode_path = "/home/mjshen/IVOS/repos/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/CWE457_s01.bc"
#repo_path = "/home/mjshen/IVOS/repos/nuttxspace/nuttx"
#sarif_path = "/home/mjshen/IVOS/OSSEmbeddedResults/apache/nuttx/7732791cd600fa9b557aecf23ecd5ef8e15359df/cpp-labeled.sarif"
ds = dataset.Dataset(repo_path, sarif_path, bitcode_path)

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
""",
        "placeholder": []
    }
    # Call variable_def_finder to find all definitions to the variable. Call get_path_constraint on the source line that defines or uses the variable.
    #print(chat_prompt.invoke(prompt_dict))
    got_final_response = False
    while not got_final_response:
        ai_msg = chain.invoke(prompt_dict)
        print(ai_msg)
        prompt_dict["placeholder"].append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            tool_str = tool_call["name"].lower()
            if tool_str == "responseformatter":
                got_final_response = True
                llm_decision = ResponseFormatter(**tool_call["args"])
            else:
                selected_tool = toolstr2callable[tool_str]
                tool_msg = selected_tool.invoke(tool_call)
                print(tool_msg)
                prompt_dict["placeholder"].append(tool_msg)

    llm_res = "LLM.BAD" if llm_decision.is_bug else "LLM.GOOD"
    print(gt, llm_res)
    print(lineno, msg, func, llm_decision.explaination, '----------------', sep='\n')

    if gt == dataset.GroundTruth.BAD and llm_decision.is_bug:
        llm_tp += 1
    elif gt == dataset.GroundTruth.BAD and not llm_decision.is_bug:
        llm_fn += 1
    elif gt == dataset.GroundTruth.GOOD and llm_decision.is_bug:
        llm_fp += 1
    elif gt == dataset.GroundTruth.GOOD and not llm_decision.is_bug:
        llm_tn += 1
print(f"LLM precision: {llm_tp}/{llm_tp+llm_fp}")
print(f"LLM recall: {llm_tp}/{llm_tp+llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
