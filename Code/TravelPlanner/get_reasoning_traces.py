import os
import json

import ast

import copy

from agent_utils import *

from Prompts.travelplanner import task, input_variables_descriptions, output_format, input_output_pairs
domain = "travelplanner"

from feedback_script.travelplanner.travelplanner_feedback_manual_cs_hc import travelplanner_feedback


from openai import OpenAI

def to_dict(obj):
    """Recursively convert an object to a dictionary."""
    if hasattr(obj, "__dict__"):
        return {key: to_dict(value) for key, value in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_dict(item) for item in obj]
    else:
        return obj

#read the contents of a jsonl file
def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]

#TODO STARTS
client = OpenAI(api_key="ENTER_DEEPSEEK_API_KEY", base_url="https://api.deepseek.com/v1")
model = "deepseek-reasoner"
name = "zeroshot_with_reasontrace"
folder_path = f"Data/TravelPlanner"
split = "train"
out_file_name = f"predictions.json"
#TODO ENDS

output_folder = f"reasoning_traces/{name}/{model}/{split}"
file_path = os.path.join(folder_path, f"{split}.jsonl")

contents = read_jsonl(file_path)

data_indices = [i for i in range(len(contents))]


data_indices = copy.deepcopy(data_indices)
data_contents = copy.deepcopy(contents)

# Create the folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

formatter_prompt = f"""You MUST STRICTLY return the OUTPUT adhering to the OUTPUT FORMAT specified:\nOUTPUT FORMAT:\n{output_format}\n"""

if os.path.exists(os.path.join(output_folder, out_file_name)):
    print("File already exists. Loading the existing file ..")
    with open(os.path.join(output_folder, out_file_name), "r") as file:
        data = json.load(file)
        data_contents = data["predictions"]

for idx, content in zip(data_indices, data_contents):
    print(f"Processing {idx} ..")
    if ("plan" not in content) or content["plan"] == None:
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

        llm_input = task + formatter_prompt + sample_outputs + convert_to_text(input_variables) + "\n OUTPUT:"

        try:
            messages = [{"role": "user", "content": llm_input}]
            response = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                timeout = 600
            )

            reasoning_content = response.choices[0].message.reasoning_content
            final_output = response.choices[0].message.content
            usage = response.usage
        except Exception as e:
            print("Error in generating reply: ", e)

        

        cost_summary = usage
        # print("USAGE COST: ", cost_summary)

        try:
            final_output = final_output.replace("```python", "").replace("```json", "").replace("```", "").strip()
            final_plan = ast.literal_eval(final_output)
        except Exception as e:
            final_plan = final_output

        content["plan"] = final_plan
        content["reasoning_content"] = reasoning_content
    
        content["cost_summary"] = to_dict(cost_summary)
        
        

        my_dict = {
            "task": task,
            "output_format": output_format,
            "output_folder_path": output_folder,
            "predictions": data_contents
        }

        # Save the output to a json file
        with open(os.path.join(output_folder, out_file_name), "w") as file:
            json.dump(my_dict, file, indent=4)

        ("##############################################") 
    else:
        print("Plan already exists for index: ", idx)

#EVALUATION
with open(os.path.join(output_folder, out_file_name), "r") as file:
    data = json.load(file)

scores, detailed_scores, detailed_feedback = travelplanner_feedback(set_type=split, tested_plans=data["predictions"], indices=data_indices)
with open(os.path.join(output_folder, out_file_name), "w") as file:
    data["scores"] = scores
    data["detailed_scores"] = detailed_scores
    data["detailed_feedback"] = detailed_feedback
    json.dump(data, file, indent=4)
      