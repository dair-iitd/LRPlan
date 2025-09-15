
import json

import autogen
from autogen import ConversableAgent
from autogen import gather_usage_summary

from EvalArena import *

import os
import re

from Prompts.timearena import *

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

def postprocess(reply):
    answer_contents = re.findall(r'<ANSWER>(.*?)</ANSWER>', reply, flags=re.DOTALL)
    answer_contents = answer_contents[0].split("\n")
    answer_contents = [x for x in answer_contents if x]
    return answer_contents

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
    
    # TODO STARTS
    agents = 3
    rounds = 2
    model = "gpt-4o"
    strategy_name = f"multiagentdebate_agents{agents}_rounds{rounds}"
    config_list = autogen.config_list_from_json("OAI_CONFIG_LIST")
    filter_dict = {"model": [model]}
    save_folder = "Outputs/TimeArena/" + strategy_name + "/" + model
    input_file_path = "Data/TimeArena"
    filename = "val.json"
    # TODO ENDS

    save_path = os.path.join(save_folder, filename)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    llm_config_list = autogen.filter_config(config_list, filter_dict)
    #Set cache_seed to None to disable caching for MAD
    llm_config_list[0]["cache_seed"] = None
    

    llm = get_agent(llm_config_list, "You are an Expert Planner.", "llm")
    plan_selector = get_agent(llm_config_list, "You are a Plan Selector.", "plan_selector")
    # response_dict = {}

    if os.path.exists(save_path):
        with open(save_path, "r") as file:
            data = json.load(file)
        data_contents = data["predictions"]
    else:
        data_contents = read_json(os.path.join(input_file_path, filename))

    for idx, content in enumerate(data_contents):
        if "plan" in content:
            print(f"Plan already exists for {idx}")
            continue

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
        final_output = postprocess(plan_selector_reply)

        print("Final Plan: ", final_output)
        content["plan"] = final_output
    
        content["cost_summary"] = cost_summary
        content["debate_rounds"] = agent_contexts
        content["plan_selector_reply"] = plan_selector_reply


        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": save_folder,
            "predictions": data_contents,
        }

        # Save the output to a json file
        with open(save_path, "w") as file:
            json.dump(my_dict, file, indent=4)
        

        # response_dict[question] = (agent_contexts, answer)

    #EVALUATION
    with open(save_path, "r") as file:
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
