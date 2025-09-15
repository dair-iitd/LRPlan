#Chain of thought code

import os


import json
import argparse

from TimeArena import *
from EvalArena import *

import re

from Agent import *

import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary

from Prompts.cot_instruction import *

def get_agent(llm_config_list, agent_system_prompt, agent_name):
    agent = ConversableAgent(
        agent_name,
        system_message=agent_system_prompt,
        llm_config={"config_list": llm_config_list},
        human_input_mode="NEVER",  # Never ask for human input.
    )
    return agent

def comma_separated_strings(string):
    return string.split(',')

def exit_action(action):
    return action.lower().strip() in ["quit", "exit"]

            
def RunData(args):

    # TODO STARTS
    model = "gpt-4o"
    strategy_name = "cot"
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    filter_dict = {"model": [model]}
    save_folder = "Outputs/TimeArena/" + strategy_name + "/" + model
    input_file_path = "Data/TimeArena"
    filename = "val.json"
    # TODO ENDS
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    save_path = os.path.join(save_folder, filename)
    llm_config_list = autogen.filter_config(config_list, filter_dict)
    
    filepath = os.path.join(input_file_path, filename)
    if os.path.exists(filepath):
        data = json.load(open(filepath, 'r'))
    else:
        raise ValueError(f"File {filepath} does not exist.")
    print("####################################")
    print(len(data), "Samples") 
    print("####################################")
    for i in range(len(data)):
        if "plan" in data[i]:
            print(f"Task {data[i]['id']} already processed.")
            continue
        else:
            query = data[i]["query"]
            prompt = COT_INSTRUCTION.replace("<<<query>>>", query)
            print(f"PROMPT:\n{prompt}")
            agent = get_agent(llm_config_list, "Your objective is to efficiently complete a task or a series of tasks as described. You must adhere to the specific requirements and constraints of each task, including dependencies and timing. Efficiency is key; complete all tasks in the shortest possible time; The key is effective parallelization of the actions.", "EfficientTasksSolver")
            if 'gpt' in model or 'o3-mini' in model or 'deepseek' in model:
                agent_reply = agent.generate_reply(messages=[{"content": prompt, "role": "user"}])
            else:
                agent_reply = agent.generate_reply(messages=[{"content": prompt, "role": "user"}])['content']
            print(f"REPLY:\n{agent_reply}")
            plan = postprocess(agent_reply)
            data[i]["plan"] = plan
            data[i]["raw_reply"] = agent_reply
            cost_summary = gather_usage_summary([agent])
            data[i]["cost"] = cost_summary
            with open(save_path, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    
    scores, detailed_scores, detailed_feedback = timearena_feedback(None, data, None, False)

    with open(save_path.replace(".json", "_scored.json"), "w") as file:
        my_dict = {
            "predictions": data,
            "scores": scores,
            "detailed_scores": detailed_scores,
            "detailed_feedback": detailed_feedback,
        }

        json.dump(my_dict, file, indent=4)
    

def postprocess(reply):
    answer_contents = re.findall(r'<ANSWER>(.*?)</ANSWER>', reply, flags=re.DOTALL)
    answer_contents = answer_contents[0].split("\n")
    answer_contents = [x for x in answer_contents if x]
    return answer_contents


def parse():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    RunData(args)