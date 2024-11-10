from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import json
import sys

from repository_manager import RepositoryManager, GroundTruth
from tools import *

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
        ("placeholder", "{history}")
    ]
)

class ResponseFormatter(BaseModel):
    """Always use this tool to structure your response to the user."""

    is_bug: bool = Field(
        description="Classification result. True for true positive, False for false positive"
    )
    explaination: str = Field(description="Explain your reasons for classification")


#model="gpt-3.5-turbo-0125"
#model="gpt-4o-mini-2024-07-18"
model="gpt-4o-2024-05-13"
llm = ChatOpenAI(temperature=0.2, model=model)
llm_with_tools = llm.bind_tools(tool_list)
structured_llm = llm.with_structured_output(ResponseFormatter)

chain = chat_prompt | llm_with_tools
chain2 = chat_prompt | structured_llm

repo_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-20241106.sarif"
bitcode_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/partial.o.bc"


repo = RepositoryManager(repo_path, sarif_path, bitcode_path)

with open("guidance.json", 'r') as json_file:
    guidance = json.load(json_file)

llm_tp, llm_fn, llm_fp, llm_tn = 0, 0, 0, 0

for srcfile, lineno, msg, func, gt, rule_id, rule_desc in repo.get_codeql_results_and_gt():
    if not guidance.get(rule_id, ''):
        continue

    prompt_dict = {
        "bug_type": rule_desc,
        "filename": srcfile,
        "lineno": lineno,
        "msg": msg,
        "func_code": func,
        "guidance": guidance[rule_id],
        "history": []
    }

    while True:
        ai_msg = chain.invoke(prompt_dict)
        #print(ai_msg)
        prompt_dict["history"].append(ai_msg)
        if not ai_msg.tool_calls:
            break
        for tool_call in ai_msg.tool_calls:
            tool_str = tool_call["name"].lower()
            print(json.dumps(tool_call, indent=2))
            selected_tool = toolstr2callable[tool_str]
            tool_msg = selected_tool.invoke(tool_call)
            print(tool_msg)
            prompt_dict["history"].append(tool_msg)
    for msg in chat_prompt.invoke(prompt_dict).to_messages():
        print(msg.pretty_print())

    llm_decision = chain2.invoke(prompt_dict)
    llm_res = "LLM.BAD" if llm_decision.is_bug else "LLM.GOOD"
    print(gt, llm_res)
    print('===========CASE END=================')
    #print(lineno, msg, func, llm_decision.explaination, '----------------', sep='\n')

    if gt == GroundTruth.BAD and llm_decision.is_bug:
        llm_tp += 1
    elif gt == GroundTruth.BAD and not llm_decision.is_bug:
        llm_fn += 1
    elif gt == GroundTruth.GOOD and llm_decision.is_bug:
        llm_fp += 1
    elif gt == GroundTruth.GOOD and not llm_decision.is_bug:
        llm_tn += 1
print(f"LLM precision: {llm_tp}/{llm_tp+llm_fp}")
print(f"LLM recall: {llm_tp}/{llm_tp+llm_fn}")
print(f"TP/FP/FN/TN: {llm_tp}/{llm_fp}/{llm_fn}/{llm_tn}")
