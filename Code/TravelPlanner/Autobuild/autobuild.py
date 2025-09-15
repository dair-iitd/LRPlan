import os
import json

import ast
import autogen

import copy
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from autogen.agentchat.contrib.society_of_mind_agent import SocietyOfMindAgent 

from agent_utils import *

from Prompts.travelplanner import task, input_variables_descriptions, output_format, input_output_pairs
domain = "travelplanner"

from feedback_script.travelplanner.travelplanner_feedback_manual_cs_hc import travelplanner_feedback

import autogen

def start_task(execution_task: str, agent_list: list, llm_config: dict, model: str):
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST", filter_dict={"model": [model]})

    group_chat = autogen.GroupChat(agents=agent_list, messages=[], max_round=12)
    manager = autogen.GroupChatManager(
        groupchat=group_chat, llm_config={"config_list": config_list, **llm_config}
    )

    user_proxy = autogen.UserProxyAgent(
        "user_proxy",
        human_input_mode="NEVER",
        code_execution_config=False,
        default_auto_reply="",
        is_termination_msg=lambda x: True,
    )

    society_of_mind_agent = SocietyOfMindAgent(
        "society_of_mind_agent",
        chat_manager= manager,
        llm_config = {"config_list": config_list, **llm_config}
    )

    final_response = user_proxy.initiate_chat(society_of_mind_agent, message=execution_task)

    return final_response, group_chat.messages

#read the contents of a jsonl file
def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

#TODO STARTS
model = "deepseek-reasoner"
name = "autobuild/"
folder_path = f"Data/TravelPlanner"
split = "validation"
output_folder = f"outputs/{name}/{model}/" 
#TODO ENDS

file_path = os.path.join(folder_path, f"{split}.jsonl")
contents = read_jsonl(file_path)
contents_for_train = read_jsonl(os.path.join(folder_path, "train.jsonl"))

train_indices = [0, 5, 10, 15, 20, 21, 25, 30, 35, 40]

validation_indices = [i for i in range(len(contents))] 

original_train_contents = [contents_for_train[i] for i in train_indices]

original_validation_contents = [contents[i] for i in validation_indices]

train_contents = copy.deepcopy(original_train_contents)
validation_contents = copy.deepcopy(original_validation_contents)

data_indices = copy.deepcopy(validation_indices)
data_contents = copy.deepcopy(validation_contents)

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# default_llm_config = {
#     'temperature': 0
# } # NOT USED IN o3-mini

default_llm_config = {}

#Agents creation
builder = AgentBuilder(config_file_or_env="OAI_CONFIG_LIST", builder_model=model, agent_model=model)

formatter_prompt = f"""You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT:\n{output_format}\n"""

training_inputs_outputs = ""
for idx, content in enumerate(train_contents):
    query = content['query']
    reference_information = eval(content['reference_information'])

    #[Additional information and preprocessing]
    attractions, accommodations, restaurants, transportation, metadata = get_reference_information_in_chunks_string(reference_information)

    query = query + "\n" + metadata
    gt_plan = ast.literal_eval(content["annotated_plan"])
    gt_plan = [d for d in gt_plan if d]
    training_inputs_outputs += f"INPUT {str(idx + 1)}: {query}\nOUTPUT {str(idx + 1)}: {gt_plan}\n\n"

building_task = task + formatter_prompt + training_inputs_outputs

print("Building Task: ", building_task)

agent_list, agent_configs = builder.build(building_task, default_llm_config, coding=True)

saved_path = builder.save(os.path.join(output_folder, "travel_agents.json"))


for idx, content in zip(data_indices, data_contents):
    if "plan" not in content:
        intermediate_outputs = dict()
        content["idx"] = idx
        query = content['query']
        # print("Query: ", query)

        reference_information = eval(content['reference_information'])

        #[Additional information and preprocessing]
        attractions, accommodations, restaurants, transportation, metadata = get_reference_information_in_chunks_string(reference_information)

        input_variables = {
            "query": query + "\n" + metadata,
            "list_of_attractions": attractions,
            "list_of_accommodations": accommodations,
            "list_of_restaurants": restaurants,
            "list_of_transportations": transportation
        }

        sample_outputs = ""
        if len(input_output_pairs) > 0:
            for index, ipop in enumerate(input_output_pairs):
                sample_outputs += f"SAMPLE OUTPUT {str(index)}: {ipop['sample_output']}\n"
        else:
            sample_outputs = ""

        final_output, group_chat_messages = start_task(
            execution_task=formatter_prompt + sample_outputs + convert_to_text(input_variables) + "\n OUTPUT:",
            agent_list=agent_list,
            llm_config=default_llm_config,
            model=model
        )

        final_output = final_output.summary

        cost_summary = autogen.gather_usage_summary(agent_list)
        print("USAGE COST: ", cost_summary)

        try:
            # final_output = final_output.replace("```python", "").replace("```json", "").replace("```", "").strip()
            final_output = extract_list_with_brackets(final_output)
            final_plan = ast.literal_eval(final_output)
            print("SUCCESS!")
        except Exception as e1:
            final_plan = final_output
                

        print("FINAL PLAN: ", final_plan)
        content["plan"] = final_plan
        content["group_chat_messages"] = group_chat_messages
    
        content["cost_summary"] = cost_summary

        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": output_folder,
            "predictions": data_contents
        }

        # Save the output to a json file
        with open(os.path.join(output_folder, f"agents_{split}_predictions" + ".json"), "w") as file:
            json.dump(my_dict, file, indent=4)

        ("##############################################")
    
    else:
        print("Plan already exists for index: ", idx)

#EVALUATION
with open(os.path.join(output_folder, f"agents_{split}_predictions" + ".json"), "r") as file:
    data = json.load(file)

scores, detailed_scores, detailed_feedback = travelplanner_feedback(set_type=split, tested_plans=data["predictions"], indices=data_indices)
with open(os.path.join(output_folder, f"agents_{split}_predictions" + ".json"), "w") as file:
    data["scores"] = scores
    data["detailed_scores"] = detailed_scores
    data["detailed_feedback"] = detailed_feedback
    json.dump(data, file, indent=4)
      