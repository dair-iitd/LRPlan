import json
import argparse

import os
from TimeArena import *

import re

from Agent import *

import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent 
from Prompts.timearena import task, input_variables_descriptions, output_format, input_output_pairs
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



def start_task(execution_task: str, agent_list: list, llm_config: dict, model: str):
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": [model]})

    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=10, role_for_select_speaker_messages="user")
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )
    agent_list[0].initiate_chat(manager, message=execution_task)

    summarizer = ConversableAgent(
        "summarizer",
        system_message="You are a summarizer agent. You are given a conversation between experts that are trying to solve the given task. Your job is to go through the conversation and output the final answer. You should only return the final answer in the required format without any additional information.",
        llm_config={"config_list": config_list},
        human_input_mode="NEVER",
    )
    gctosum = "Task: " + execution_task + "\n\n"
    gctosum += "Conversation: \n\n" + str(group_chat.messages) + "\n\n" + "Final Answer: "
    
    final_response = summarizer.generate_reply(
        messages=[{"content": gctosum, "role": "user"}]
    )

    return final_response, group_chat.messages
   
def is_timearena_correct(final_results):
    return all(task_info['fully_completed'] for task_info in final_results.values())

def RunData(args):

    # TODO STARTS
    model = "gpt-4o"
    strategy_name = "autobuild"
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    filter_dict = {"model": [model]}
    save_folder = "Outputs/TimeArena/" + strategy_name + "/" + model
    input_file_path = "Data/TimeArena"
    filename = "val.json"
    # TODO ENDS

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    llm_config_list = autogen.filter_config(config_list, filter_dict)
    builder = AgentBuilder(config_file_or_env="OAI_CONFIG_LIST", builder_model=model, agent_model=model, max_agents=5)
    # default_llm_config = {
    #     'temperature': 0
    # }
    default_llm_config = {}
    filepath = os.path.join(input_file_path, filename)
    if os.path.exists(filepath):
        data = json.load(open(filepath, 'r'))
    else:
        raise ValueError(f"File {filepath} does not exist.")
    if os.path.exists(os.path.join(save_folder, "agents.json")):
        agent_list, agent_configs = builder.load(os.path.join(save_folder, "agents.json"))
    else:
    
        train_contents_path = input_file_path + "/train.json"
        with open(train_contents_path, "r") as file:
            train_contents = json.load(file)
        
        formatter_prompt = f"""You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT:\n{output_format}\n"""
        
        
        # 9 samples as building task
        building_task = task + formatter_prompt
        zidx = 1
        for i in [0, 1, 2, 9, 10, 11, 18, 19, 20, 21]:
            query = train_contents[i]["query"]
            gt_plan = train_contents[i]["oracle_plan"]
            building_task += f"INPUT {str(zidx)}: {query}\nOUTPUT {str(zidx)}: {list(gt_plan)}\n\n"
            zidx += 1
        
        agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=False)
        saved_path = builder.save(os.path.join(save_folder, "agents.json"))

    print("####################################")
    print(len(data), "Samples") 
    print("####################################")
    for i in range(len(data)):
        if "plan" in data[i]:
            print(f"Task {data[i]['id']} already processed.")
            continue
        else:
            prompt = data[i]["query"]
        
            final_output, gc = start_task(
            execution_task=prompt,
            agent_list=agent_list,
            llm_config=default_llm_config,
            model=model
            )
            try:
                plan = postprocess(final_output.summary)
            except:
                plan = []
            print(f"PLAN GENERATED:\n{plan}")
            data[i]["plan"] = plan
            data[i]["raw_reply"] = final_output.summary
            data[i]["groupchat"] = gc
            data[i]["socofminds"] = final_output.chat_history
            data[i]["cost"] = final_output.cost
            cost_summary = gather_usage_summary(agent_list)
            data[i]["cost_agents"] = cost_summary
            with open(filepath, 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
    

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