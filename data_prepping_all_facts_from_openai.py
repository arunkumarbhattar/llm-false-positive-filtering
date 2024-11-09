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
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import dataset
from prompts import *

from tools import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

# Initialize the LLM
model = "gpt-4o-2024-05-13"  # Replace with your actual model name
llm = ChatOpenAI(temperature=0.2, model=model)

with open("guidance.json", 'r') as json_file:
    guidance = json.load(json_file)

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

# Main loop to process each CodeQL alert
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
