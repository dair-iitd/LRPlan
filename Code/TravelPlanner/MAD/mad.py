from glob import glob
import pandas as pd
import json

import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary
import ast

from Prompts.travelplanner import *

import os
import re

from agent_utils import *
from feedback_script.travelplanner.travelplanner_feedback_manual_cs_hc import travelplanner_feedback

#read the contents of a jsonl file
def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

def construct_message(agents, question, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your plan is correct. Put your final plan at the end of your response."}

    prefix_string = "These are the plans to the query from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent solution: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + """\n\n Using the reasoning from other agents as additional advice, can you give an updated plan? Examine your plan and that other agents step by step. Put your final plan at the end of your response.""".format(question)
    return {"role": "user", "content": prefix_string}


def construct_assistant_message(completion):
    # content = completion["choices"][0]["message"]["content"]
    content = completion
    return {"role": "assistant", "content": content}

def get_agent(llm_config_list, agent_system_prompt, agent_name):
    agent = ConversableAgent(
        agent_name,
        system_message=agent_system_prompt,
        llm_config={"config_list": llm_config_list},
        human_input_mode="NEVER",  # Never ask for human input.
    )
    return agent

def generate_answer(llm, answer_context, model = "gpt-4o-mini"):
    
    try:
        # completion = openai.ChatCompletion.create(
        #           model=model,
        #           messages=answer_context,
        #           n=1)
        reply = llm.generate_reply(messages=answer_context, n=1)
    except Exception as e:
        print(f"error {e}......")
        return None

    return reply

if __name__ == "__main__":
    
    #TODO STARTS
    agents = 3
    rounds = 2
    model = "deepseek-chat"
    split = "validation"
    strategy_name = f"multiagentdebate_agents{agents}_rounds{rounds}"
    folder_path = f"Data/TravelPlanner"
    output_folder = f"outputs/{strategy_name}/{model}/"
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    #TODO ENDS

    input_file_path = os.path.join(folder_path, split + ".jsonl")
    save_path = os.path.join(output_folder, split + ".jsonl")
    filter_dict = {"model": [model]}

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    
    llm_config_list = autogen.filter_config(config_list, filter_dict)
    llm_config_list[0]["cache_seed"] = None
    print("LLM Config List: ", llm_config_list)


    llm = get_agent(llm_config_list, "You are an Expert Planner.", "llm")
    plan_selector = get_agent(llm_config_list, "You are a Plan Selector.", "plan_selector")
    # response_dict = {}

    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            data = json.load(file)
        data_contents = data["predictions"]
    else:
        data_contents = read_jsonl(input_file_path)

    data_indices = [i for i in range(len(data_contents))]

    for idx, content in enumerate(data_contents):
        if "plan" in content:
            print(f"Plan already exists for {idx}")
            continue
        print(f"Processing task {idx}: ")

        content["idx"] = idx
        query = content["query"]
        print("Query: ", query)

        reference_information = eval(content['reference_information'])

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

        agent_contexts = [[{"role": "user", "content": str(input_variables)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                    message = construct_message(agent_contexts_other, input_variables, 2 * round - 1)
                    agent_context.append(message)

                completion = generate_answer(llm, agent_context, model)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)

        #Get single plan from the last round
        message = construct_message(agent_contexts_other, input_variables, 2 * rounds - 1)

        plan_selector_context = f"""
        {str(input_variables)}\n
        {message}\n
        You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT: {output_format}\n
        {sample_outputs}
        """

        plan_selector_reply = plan_selector.generate_reply(messages=[{"content": plan_selector_context, "role": "user"}], n=1)
        print("[Plan Selector Reply]: ", plan_selector_reply)

        cost_summary = autogen.gather_usage_summary([llm, plan_selector])
        print("USAGE COST: ", cost_summary)
        
        try:
            final_output = extract_list_with_brackets(plan_selector_reply)
            final_output = ast.literal_eval(final_output)
        except:
            final_output = plan_selector_reply

        print("Final Plan: ", final_output)
        content["plan"] = final_output
    
        content["cost_summary"] = cost_summary
        content["debate_rounds"] = agent_contexts
        content["plan_selector_reply"] = plan_selector_reply


        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": output_folder,
            "predictions": data_contents,
        }

        # Save the output to a json file
        with open(save_path, "w") as file:
            json.dump(my_dict, file, indent=4)
        

    #EVALUATION
    with open(save_path, "r") as file:
        data = json.load(file)

    if "scores" not in data:
        scores, detailed_scores, detailed_feedback = travelplanner_feedback(set_type=split, tested_plans=data["predictions"], indices=data_indices)
        with open(save_path.replace(".json", "_scored.json"), "w") as file:
            data["scores"] = scores
            data["detailed_scores"] = detailed_scores
            data["detailed_feedback"] = detailed_feedback
            json.dump(data, file, indent=4)
    else:
        print("Scores already exists")
