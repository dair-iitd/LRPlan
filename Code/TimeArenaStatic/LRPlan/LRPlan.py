#LR Plan 
import os

import copy
import json

import argparse

from TimeArena import *
from EvalArena import *
from agent_utils import *

import re

from Agent import *
import pdb
import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary


from Prompts.pattern_extractor_corrector import *

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

def is_timearena_correct(final_results):
    return all(task_info['fully_completed'] for task_info in final_results.values())


def exit_action(action):
    return action.lower().strip() in ["quit", "exit"]

def read_json(file_path):
    """
    Reads a JSON file and returns the data.
    
    Args:
        file_path (str): Path to the JSON file.
    
    Returns:
        data (dict or list): Parsed JSON data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

            
def RunData(args):
    # TODO STARTS
    model = "deepseek-chat"
    model2 = "deepseek-reasoner"
    strategy_name = "LRPlan"
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    save_folder = "Outputs/TimeArena/" + strategy_name + "/" + model
    input_file_path = "Data/TimeArena"
    filename = "val.json"
    NUM_SAMPLES = 2 # Number of positive or negative samples to use for training
    reasoning_file = "Code/TimeArenaStatic/reasoning_traces/zeroshot_with_reasontrace/deepseek-reasoner/train/predictions.json"
    # TODO ENDS

    filter_dict = {"model": [model]}
    llm_config_list = autogen.filter_config(config_list, filter_dict)
    filter_dict2 = {"model": [model2]}
    llm_config_list2 = autogen.filter_config(config_list, filter_dict2)

    #Added for DeepSeek Reasoner
    llm_config_list2[0]["timeout"] = 1800


    with open(reasoning_file, "r") as file:
        reasoning_contents = json.load(file)
    
    data_contents = read_json(os.path.join(input_file_path, filename))


    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
   
    save_path = os.path.join(save_folder, filename)
    
    default_llm_config = {
        'temperature': 0
    }

    formatter_prompt = f"""You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT:\n{output_format}\n"""

    correct_predictions = []
    wrong_predictions = []
    idx = 1
    for content, detailed_feedback in zip(reasoning_contents["predictions"], reasoning_contents["detailed_feedback"]):
        assert content["id"] == detailed_feedback["id"]
        query = content['query']
        reasoning_content = content["reasoning_content"]

        final_results = detailed_feedback["detailed_evaluation"]["final_results"]
        correct = is_timearena_correct(final_results)
        if final_results != {} and correct == True:
            correct_predictions.append({
            "id": content["id"],
            "tasks": content["tasks"],
            "query": query,
            "reasoning_content": reasoning_content
        })
        else:
            wrong_predictions.append({
            "id": content["id"],
            "tasks": content["tasks"],
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


    for idx, content in enumerate(data_contents):
        print(f"Processing task {idx}: ", content["tasks"])
        
        idx = content["id"]
        query = content["query"]
        print("Query: ", query)

        input_variables = {
            "query": query
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

        print("[PLANNER EXPERT REPLY]:", planner_output)

        refiner_prefix_variables = f"""
        ###PLANNER EXPERT REPLY: {planner_output}\n
        ###SELF-CORRECTION INSIGHTS: {self_corrector_reply}\n
        """

        refiner_output = refiner.generate_reply(messages=[{"content": refiner_prefix_prompt + refiner_prefix_variables +formatter_prompt + sample_outputs + convert_to_text(input_variables) + "\n OUTPUT:", "role": "user"}])

        print("[REFINER EXPERT REPLY]:", refiner_output)

        content["prompts"] = {
            "planner_prompt": planner_prefix_prompt + planner_prefix_variables + "<<<convert_to_text(input_variables)>>>" + "\n OUTPUT:",
            "refiner_prompt": refiner_prefix_prompt + refiner_prefix_variables + "<<<convert_to_text(input_variables)>>>" + "\n OUTPUT:"
        }
        
        content["intermediate_outputs"] = {
            "planner": planner_output,
            "refiner": refiner_output
        }

        final_output = refiner_output
        final_plan = postprocess(final_output)

        cost_summary = autogen.gather_usage_summary([planner, refiner, pattern_recognizer, rule_extractor, self_corrector])
        print("USAGE COST: ", cost_summary)

        print("Final Plan: ", final_output)


        content["plan"] = final_plan
    
        content["cost_summary"] = cost_summary

        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": save_folder,
            "predictions": data_contents,
            "agent_replies": {
                "pattern_recognizer": pattern_recognizer_reply,
                "rule_extractor": rule_extractor_reply,
                "self_corrector": self_corrector_reply,
            }
        }

        # Save the output to a json file
        with open(os.path.join(save_folder, filename), "w") as file:
            json.dump(my_dict, file, indent=4)


        ("##############################################")
    
    else:
        print("Plan already exists for index: ", idx)

    #EVALUATION
    with open(os.path.join(save_folder, filename), "r") as file:
        data = json.load(file)

    if "scores" not in data:
        scores, detailed_scores, detailed_feedback = timearena_feedback(set_type=None, plans=data["predictions"], indices=None, llm_config_list=llm_config_list, llm_feedback_flag=False)
        with open(save_path.replace(".json", "_scored.json"), "w") as file:
            data["scores"] = scores
            data["detailed_scores"] = detailed_scores
            data["detailed_feedback"] = detailed_feedback
            json.dump(data, file, indent=4)
    else:
        print("Scores already exists")   


def postprocess(reply):
    try:
        answer_contents = re.findall(r'<ANSWER>(.*?)</ANSWER>', reply, flags=re.DOTALL)
        answer_contents = answer_contents[0].split("\n")
        answer_contents = [x for x in answer_contents if x]
        return answer_contents
    except Exception:
        return reply 


def parse():
    parser = argparse.ArgumentParser()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse()

    RunData(args)