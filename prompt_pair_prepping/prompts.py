from repository_manager import RepositoryManager, GroundTruth

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

# Initialize the dataset
repo_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3"
sarif_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/cpp-labeled-20241106.sarif"
bitcode_path = "/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/partial.o.bc"

repo = RepositoryManager(repo_path, sarif_path, bitcode_path)
