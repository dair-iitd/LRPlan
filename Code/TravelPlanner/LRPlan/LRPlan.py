#LR Plan

import os
import json

import ast
import autogen

import copy

from autogen import ConversableAgent

from agent_utils import *

from Prompts.travelplanner import task, input_variables_descriptions, output_format, input_output_pairs
domain = "travelplanner"

from feedback_script.travelplanner.travelplanner_feedback_manual_cs_hc import travelplanner_feedback

import autogen
from Prompts.pattern_extractor_corrector import *


#read the contents of a jsonl file
def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

#TODO STARTS
model = "gpt-4o"
model2 = "o3-mini"
name = "LRPlan"
split = "validation" 
folder_path = f"Data/TravelPlanner"
reasoning_file = "Code/TravelPlanner/reasoning_traces/zeroshot_with_reasontrace/deepseek-reasoner/train/predictions.json"
output_folder = f"outputs/{name}/{model}/" 
config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
NUM_SAMPLES = 5 # Number of positive or negative samples to use for training
#TODO ENDS

filter_dict = {"model": [model]}
llm_config_list = autogen.filter_config(config_list, filter_dict)
filter_dict2 = {"model": [model2]}
llm_config_list2 = autogen.filter_config(config_list, filter_dict2)

file_path = os.path.join(folder_path, f"{split}.jsonl")


with open(reasoning_file, 'r') as file:
    reasoning_contents = json.load(file)

contents = read_jsonl(file_path)

data_indices = [i for i in range(len(contents))] 

data_contents = copy.deepcopy(contents)

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

default_llm_config = {
    'temperature': 0
} # NOT USED IN o3-mini

# default_llm_config = {}


formatter_prompt = f"""You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT:\n{output_format}\n"""

# Cumulating correct predictions
correct_predictions = []
wrong_predictions = []
r_predictions = reasoning_contents["predictions"]
r_detailed_feedback = reasoning_contents["detailed_feedback"]
for r_pred, r_dfeed in zip(r_predictions, r_detailed_feedback):
    assert r_pred["idx"] == r_dfeed["idx"]
    query = r_dfeed['query']
    reasoning_content = r_pred["reasoning_content"]
    print("Query: ", query)
    print("Reasoning Content: ", reasoning_content)
    if (is_travelplanner_correct(r_dfeed["detailed_evaluation"]) == True):
        correct_predictions.append({
            "idx": r_pred["idx"],
            "query": query,
            "reasoning_content": reasoning_content
        })
    else:
        wrong_predictions.append({
            "idx": r_pred["idx"],
            "query": query,
            "reasoning_content": reasoning_content
        })

training_inputs_outputs = ""
zidx = 1
for t_dict in correct_predictions[:NUM_SAMPLES]:
    training_inputs_outputs += f"SAMPLE INPUT #{zidx}:\n{t_dict['query']}\nSAMPLE REASONING TRACE #{zidx}:\n{t_dict['reasoning_content']}\nVERDICT: Correct\n\n"
    zidx += 1

for t_dict in wrong_predictions[:NUM_SAMPLES]:
    training_inputs_outputs += f"SAMPLE INPUT #{zidx}:\n{t_dict['query']}\nSAMPLE REASONING TRACE #{zidx}:\n{t_dict['reasoning_content']}\nVERDICT: Incorrect\n\n"
    zidx += 1


building_task = f"""
TASK DESCRIPTION: {task}\n
OUTPUT FORMAT: {output_format}\n
{training_inputs_outputs}
"""

print("Building Task: ", building_task)
# print("Total Correct Predictions: ", len(correct_predictions))

pattern_recognizer = get_agent(llm_config_list, PATTERN_RECOGNIZER_SYSTEM_PROMPT, "pattern-recognizer")
rule_extractor = get_agent(llm_config_list, RULE_EXTRACTOR_SYSTEM_PROMPT, "rule-extractor")
self_corrector = get_agent(llm_config_list, SELF_CORRECTOR_SYSTEM_PROMPT, "self-corrector")

pattern_recognizer_prompt = PATTERN_RECOGNIZER_PROMPT.replace("<<<sample_input_reasoning_trace>>>", building_task)
pattern_recognizer_reply = pattern_recognizer.generate_reply(messages=[{"content": pattern_recognizer_prompt, "role": "user"}])
print("[PATTERN RECOGNIZER REPLY]:", pattern_recognizer_reply)

self_corrector_prompt = SELF_CORRECTOR_PROMPT.replace("<<<sample_input_reasoning_trace>>>", building_task)
self_corrector_reply = self_corrector.generate_reply(messages=[{"content": self_corrector_prompt, "role": "user"}])
print("[SELF CORRECTOR REPLY]:", self_corrector_reply)

rule_extractor_prompt = RULE_EXTRACTOR_PROMPT.replace("<<<identified_research_patterns>>>", pattern_recognizer_reply)
rule_extractor_reply = rule_extractor.generate_reply(messages=[{"content": rule_extractor_prompt, "role": "user"}])
print("[RULE EXTRACTOR REPLY]:", rule_extractor_reply)

planner = get_agent(llm_config_list, "You are a Planner Expert.", "planner")
refiner = get_agent(llm_config_list2, "You are a Refiner Expert.", "refiner")


planner_prefix_prompt = f"""
You are a Planner Expert. You will be given a task description, output format, task, identified reasoning patterns and heuristics. You MUST generate a PLAN adhering to the output format specified.\n
"""

refiner_prefix_prompt = f"""
You are a Refiner Expert. Make use of the self correction insights and the plan generated by the Planner Expert. You need to carefully analyze the plan generated by the Planner Expert and refine it to make it more accurate. Refinement needs to be done adhering to the output format specified. Refinement must be done only if neccessary.\n
"""


for idx, content in zip(data_indices, data_contents):
    if "plan" not in content:
        intermediate_outputs = dict()
        content["idx"] = idx
        query = content['query']
        print("Query: ", query)

        reference_information = eval(content['reference_information'])

        #[Additional information and preprocessing]
        attractions, accommodations, restaurants, transportation, metadata = get_reference_information_in_chunks_jsonified(reference_information)


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


        planner_prefix_variables = f"""
        ###TASK DESCRIPTION: {task}\n
        ###OUTPUT FORMAT: {output_format}\n
        ###IDENTIFIED REASONING PATTERNS: {pattern_recognizer_reply}\n
        ###HEURISTICS: {rule_extractor_reply}\n"""

        planner_output = planner.generate_reply(messages=[{"content": planner_prefix_prompt + planner_prefix_variables + formatter_prompt + sample_outputs + convert_to_text(input_variables) + "\n OUTPUT:", "role": "user"}])

        refiner_prefix_variables = f"""
        ###PLANNER EXPERT REPLY: {planner_output}\n
        ###SELF-CORRECTION INSIGHTS: {self_corrector_reply}\n
        """

        refiner_output = refiner.generate_reply(messages=[{"content": refiner_prefix_prompt + refiner_prefix_variables +formatter_prompt + sample_outputs + convert_to_text(input_variables) + "\n OUTPUT:", "role": "user"}])

        content["prompts"] = {
            "planner_prompt": planner_prefix_prompt + planner_prefix_variables + "formatter_prompt + sample_outputs + <<<convert_to_text(input_variables)>>>" + "\n OUTPUT:",
            "refiner_prompt": refiner_prefix_prompt + refiner_prefix_variables + "formatter_prompt + sample_outputs + <<<convert_to_text(input_variables)>>>" + "\n OUTPUT:"
        }
        
        content["intermediate_outputs"] = {
            "planner": planner_output,
            "refiner": refiner_output
        }

        final_output = refiner_output
        
        cost_summary = autogen.gather_usage_summary([planner, refiner, pattern_recognizer, rule_extractor, self_corrector])
        print("USAGE COST: ", cost_summary)

        try:
            # final_output = final_output.replace("```python", "").replace("```json", "").replace("```", "").strip()
            final_output = extract_list_with_brackets(final_output)
            final_plan = ast.literal_eval(final_output)
            print("SUCCESS!")
        except Exception as e:
            final_plan = final_output

        content["plan"] = final_plan

        print("Final Plan: ", final_plan)
        print("$"*50)
        
    
        content["cost_summary"] = cost_summary
        

        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": output_folder,
            "predictions": data_contents,
            "agent_replies": {
                "pattern_recognizer": pattern_recognizer_reply,
                "rule_extractor": rule_extractor_reply,
                "self_corrector": self_corrector_reply,
            }
        }

        # Save the output to a json file
        with open(os.path.join(output_folder, f"agents_predictions_{split}" + ".json"), "w") as file:
            json.dump(my_dict, file, indent=4)

        ("##############################################")
    
    else:
        print("Plan already exists for index: ", idx)

#EVALUATION
with open(os.path.join(output_folder, f"agents_predictions_{split}" + ".json"), "r") as file:
    data = json.load(file)

scores, detailed_scores, detailed_feedback = travelplanner_feedback(set_type=split, tested_plans=data["predictions"], indices=data_indices)
with open(os.path.join(output_folder, f"agents_predictions_{split}" + ".json"), "w") as file:
    data["scores"] = scores
    data["detailed_scores"] = detailed_scores
    data["detailed_feedback"] = detailed_feedback
    json.dump(data, file, indent=4)

