import logging
import re
import json
import argparse
import os

from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

from tools import *  # Assumes tool_list is defined here
from repository_manager import RepositoryManager, GroundTruth
import tools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data by processing CodeQL alerts using OpenAI API."
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        required=True,
        help="OpenAI API key."
    )
    parser.add_argument(
        "--repo_path",
        type=str,
        required=True,
        help="Path to the Juliet test suite repository."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="fine_tuning_training_data.jsonl",
        help="Output JSONL file name."
    )
    args = parser.parse_args()

    openai_api_key = args.openai_api_key
    repo_path = args.repo_path
    output_file = args.output_file

    # Initialize the LLM
    # Choose the appropriate model
    MODEL_NAME = "gpt-4o-2024-05-13"
    # MODEL_NAME = "gpt-3.5-turbo-0125"

    llm = ChatOpenAI(
        temperature=0.2,
        model=MODEL_NAME,
        openai_api_key=openai_api_key
    )

    # Load guidance data
    guidance_file = "guidance.json"
    with open(guidance_file, 'r') as json_file:
        guidance = json.load(json_file)
    logger.info("Guidance data successfully loaded.")

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

1. **Tool Selection**: Plan which tools to invoke to assist in triaging this bug. Do not execute any tool. You might need 1 or more tools at the same time. Provide your answer in a comma separated format

2. **Reasoning**: Provide the reasoning behind your selection in plain English, using a chain-of-thought format.
""".strip()

    agent_kwargs = {
        "system_message": system_message
    }

    # Initialize the agent with the tools and system message, but prevent tool execution
    agent = initialize_agent(
        tools=tool_list,
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

    # Define the SARIF subpaths to process
    sarif_subpaths = [
        "C/testcases/CWE457_Use_of_Uninitialized_Variable/s02/",
        "C/testcases/CWE121_Stack_Based_Buffer_Overflow/s01/"
    ]

    # Iterate over each SARIF subpath
    for subpath in sarif_subpaths:
        sarif_full_path = os.path.join(repo_path, subpath, "cpp-labeled.sarif")
        logger.info(f"Processing SARIF file: {sarif_full_path}")

        if not os.path.isfile(sarif_full_path):
            logger.warning(f"SARIF file not found at {sarif_full_path}. Skipping this subpath.")
            continue

        # Construct bitcode path
        bitcode_path = os.path.join(repo_path, subpath, "partial.o.bc")

        # Initialize the dataset
        repo = RepositoryManager(repo_path, sarif_full_path, bitcode_path)
        tools.repo = repo  # Assign to tools if required

        # Main loop to process each CodeQL alert
        for srcfile, lineno, msg, func, gt, rule_id, rule_desc in repo.get_codeql_results_and_gt():
            if not guidance.get(rule_id, ''):
                logger.warning(f"No guidance found for rule ID: {rule_id}. Skipping this alert.")
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
""".strip()
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
    try:
        with open(output_file, 'w') as f:
            for example in data:
                json.dump(example, f)
                f.write('\n')
        logger.info(f"Training data saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")

if __name__ == "__main__":
    main()
