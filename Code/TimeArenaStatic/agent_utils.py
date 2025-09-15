import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary
import re
import json


def remove_think_tags(text):
    # Find all content between <think> and </think> tags
    think_contents = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    
    # Remove the <think> tags along with their content
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    
    return cleaned_text, think_contents

def extract_from_to_on(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B on <date>" in the given text
    
    Args:
    - text (str): The input string.
    
    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r'from\s(.*?)\sto\s(.*?)\son'
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

def extract_from_to(text: str):
    """
    Extracts 'A' and 'B' from the format "from A to B" in the given text
    
    Args:
    - text (str): The input string.
    
    Returns:
    - tuple: A tuple containing 'A' and 'B'. If no match is found, returns (None, None).
    """
    pattern = r"from\s+(.+?)\s+to\s+([^,]+)(?=[,\s]|$)"
    matches = re.search(pattern, text)
    return matches.groups() if matches else (None, None)

def extract_date(text: str):
    """
    Extracts the date from the given text.
    
    Args:
    - text (str): The input string.
    
    Returns:
    - str: The extracted date.
    """
    date_pattern = r'\d{4}-\d{2}-\d{2}'

    # Use re.search() to find the first match for the date pattern
    match = re.search(date_pattern, text)

    return match.group(0) if match else None

def extract_dict(text):
    dict_pattern = r"\{.*?\}"

    matches = re.findall(dict_pattern, text, re.DOTALL)

    if matches:
        dict_string = matches[0]
        return dict_string
    
    return text

def extract_python_dict2(text):
    # Define a regex pattern to capture content between ```python and ```
    #USEFUL FOR CODE where indentation is important
    pattern = r'```python(.*?)```'
    
    # Use re.DOTALL to allow '.' to match newlines
    matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        return text

    return matches[0]

def get_text_between_start_and_end(text):
    if "<start>" in text and "<end>" in text:
        return text.split("<start>")[1].split("<end>")[0]
    else:
        print("!!!Start and End not found in the text.")
        return text

def get_considered_instances(split_file, split):
    with open(split_file, 'r') as file:
        split_data = json.load(file)

    split_instances = split_data[split]

    return split_instances

def convert_to_text(inputs):
    prompt = ""
    for key, value in inputs.items():
        prompt += f"{str(key)}: {str(value)}\n"
    return prompt

def convert_ipop_to_prompt(sample_input_output_pairs):
    prompt = ""
    for sample in sample_input_output_pairs.values():
        
        if "input" in sample and sample["input"] != "":
            prompt += f"SAMPLE INPUT:\n{sample['input']}\n"
        if "output" in sample and sample["output"] != "":
            prompt += f"SAMPLE OUTPUT:\n{sample['output']}\n"

    prompt += "Provide the output for the following input:\n"

    return prompt

def create_agents(agents_descriptions, llm_config_list):
    agents = {}
    assert type(agents_descriptions) == dict
    for agent_name, (_, agent_system_message) in agents_descriptions.items():
        # print(agent_name)
        # print(agent_system_message)
        new_agent = ConversableAgent(
            agent_name,
            system_message= agent_system_message,
            llm_config={"config_list": llm_config_list},
            human_input_mode="NEVER",  # Never ask for human input.
        )
        agents[agent_name] = new_agent
    
    return agents

def add_agent_reply_placeholders(input_variables, agents_descriptions):
    inputs = input_variables.copy()
    for agent_name in agents_descriptions.keys():
        inputs[agent_name+"_reply"] = []

    return inputs

def get_current_action(orchestrator_reply, DAG):
    status = (False, "No valid action found. Please select from the actions specified.")
    for action in DAG.keys():
        if action.lower() == orchestrator_reply.lower():
            status = (True, action)
            break

    return status

def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def get_agent(llm_config_list, agent_system_prompt, agent_name):
    agent = ConversableAgent(
        agent_name,
        system_message=agent_system_prompt,
        llm_config={"config_list": llm_config_list},
        human_input_mode="NEVER",  # Never ask for human input.
    )
    return agent

def get_agents_creator_prompt(agents_creation_prompt, task, input_variables_description, output_format, input_output_pairs):
    """
    <task_description>
    <input_variables_description>
    <output_format>
    <sample_input_1>
    <sample_output_1>
    <sample_input_2>
    <sample_output_2>
    ...
    """
    prompt = f"""
    TASK DESCRIPTION: {task}\n
    INPUT VARIABLES DESCRIPTION: {input_variables_description}\n
    {output_format}\n
    """

    for idx, input_output_pair in enumerate(input_output_pairs, start=1):
        expanded_sample_input = convert_to_text(input_output_pair["sample_input"])
        prompt += f"""
        SAMPLE INPUT {str(idx)}: {expanded_sample_input}\n
        SAMPLE OUTPUT {str(idx)}: {input_output_pair["sample_output"]}\n
        """
    
    return agents_creation_prompt.replace("<<<task_sample_input_output>>>", prompt)

def get_gradient_computer_prompt(gradient_computation_prompt, previous_agents, detailed_feedback):
    """
    <previous_agents>
    <feedback_over_training_set>
    """
    prompt = ""
    for feedback in detailed_feedback:
        if feedback["plan"] is not None:

            prompt += f"""
            INPUT: {feedback["query"]}\n
            OUTPUT: {feedback["plan"]}\n
            FEEDBACK: {feedback["llm_feedback"]}\n
            """

    return gradient_computation_prompt.replace("<<<agents_and_roles>>>", str(previous_agents)).replace("<<<feedback_over_training_set>>>", prompt)

def get_agents_updater_prompt(agents_updation_prompt, previous_agents, agent_specific_feedback):
    return agents_updation_prompt.replace("<<<agents_and_roles>>>", str(previous_agents)).replace("<<<agent_specific_feedback>>>", str(agent_specific_feedback))
    


def get_dag_creator_prompt(dag_creation_prompt, agents_creator_reply, input_variables_descriptions):
    prompt = f"""
    VARIABLES: {str(input_variables_descriptions)}\n
    AGENTS: {str(agents_creator_reply)}\n
    """
    # print("[DAG CREATOR PROMPT]:", prompt)
    return dag_creation_prompt.replace("<<<agents_and_roles>>>", prompt)


#LLM Feedback
def get_feedback_prompt(query, plan, detailed_evaluation_final_results):
    prompt = f""" You are given the following query, generated plan and the validator response of the plan. Please provide a detailed reasoning for the completeness and feasibility of the plan. Carefully analyze the plan to provide feedback. Reply 'NA' if the verdict is correct. \n\n
    QUERY: {query}\nPLAN: {plan}\nVALIDATOR RESPONSE: {detailed_evaluation_final_results}\n FEEDBACK: """
    return prompt

def get_llm_feedback(query, plan, correct, llm_config_list):
    feedback_engine = ConversableAgent(
        "plan-feedback",
        system_message="You are an expert in planning. Please provide criticism on the generated plan. Go through the plan in detail and check for any inconsistencies. Be concise and clear in your output.",
        llm_config={"config_list": llm_config_list},
        human_input_mode="NEVER",  # Never ask for human input.
    )

    feedback_engine_prompt = get_feedback_prompt(query, plan, correct)

    print("[FEEDBACK PROMPT]: ", feedback_engine_prompt)

    feedback = feedback_engine.generate_reply(messages=[{"content": feedback_engine_prompt, "role": "user"}])

    print("[FEEDBACK REPLY]: ", feedback)

    feedback_cost = gather_usage_summary([feedback_engine])

    return feedback, feedback_cost