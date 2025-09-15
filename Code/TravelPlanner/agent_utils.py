import re
import json
import pandas as pd
import numpy as np
import sys
import os
from autogen import ConversableAgent

## To add the path to the folder outside the current directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from tools.flights.apis import Flights
from tools.accommodations.apis import Accommodations
from tools.restaurants.apis import Restaurants
from tools.googleDistanceMatrix.apis import GoogleDistanceMatrix
from tools.attractions.apis import Attractions
from collections import Counter

FLIGHT_DB_PATH = "Code/TravelPlanner/feedback_script/travelplanner/database/flights/clean_Flights_2022.csv"
ACCOMMODATION_DB_PATH = "Code/TravelPlanner/feedback_script/travelplanner/database/accommodations/clean_accommodations_2022.csv"
RESTAURANTS_DB_PATH = "Code/TravelPlanner/feedback_script/travelplanner/database/restaurants/clean_restaurant_2022.csv"
ATTRACTIONS_DB_PATH = "Code/TravelPlanner/feedback_script/travelplanner/database/attractions/attractions.csv"

flight = Flights(path=FLIGHT_DB_PATH)
accommodation = Accommodations(path=ACCOMMODATION_DB_PATH)
restaurants = Restaurants(path=RESTAURANTS_DB_PATH)
googleDistanceMatrix = GoogleDistanceMatrix()
attractions = Attractions(path=ATTRACTIONS_DB_PATH)

#read the contents of a jsonl file
def read_jsonl(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]
    
def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
    
def dataframe_to_list_of_dicts(df: pd.DataFrame) -> list:
    """
    Converts a pandas DataFrame into a list of dictionaries.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        list: A list where each item is a dictionary representing a row in the dataframe.
    """
    return df.to_dict(orient='records')

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

def get_agent(llm_config_list, agent_system_prompt, agent_name):
    agent = ConversableAgent(
        agent_name,
        system_message=agent_system_prompt,
        llm_config={"config_list": llm_config_list},
        human_input_mode="NEVER",  # Never ask for human input.
    )
    return agent

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

def check_travelplanner_keys(data, required_keys = ["days", "current_city", "transportation", "breakfast", "attraction", "lunch", "dinner", "accommodation"]):
    """Check if all required keys are present in the dictionary."""
    return all(key in data for key in required_keys)

def extract_list_with_brackets(text):
    # Regex pattern to capture content inside [ and ], including the brackets
    matches = re.findall(r'(\[.*?\])', text, re.DOTALL)
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



def get_reference_information_in_chunks(reference_information):
    """
    Splits the reference_information into attractions, accommodations, restaurants, and transportation.
    """
    attractions_lst = []
    accommodations_lst = []
    restaurants_lst = []
    transportation_lst = []
    transportation_metadata = "Information available about "
    for pair in reference_information:
        description = pair['Description']
        if "attractions" in description.lower():
            city = description.split("Attractions in ")[1]
            info = attractions.run(city)
            attractions_lst.append({"Description": description, "Information": info})

        if "accommodations" in description.lower():
            city = description.split("Accommodations in ")[1]
            info = accommodation.run(city)
            accommodations_lst.append({"Description": description, "Information": info})

        if "restaurants" in description.lower():
            city = description.split("Restaurants in ")[1]
            info = restaurants.run(city)
            restaurants_lst.append({"Description": description, "Information": info})

        if "flight" in description.lower():
            org_city, dest_city = extract_from_to_on(description)
            date = extract_date(description)
            info = flight.run(org_city, dest_city, date)
            transportation_metadata += description + ", "
            transportation_lst.append({"Description": description, "Information": info})

        if "driving" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='self-driving')
            # transportation_metadata += description + ", " # No need to add this to the metadata
            transportation_lst.append({"Description": description, "Information": info})
        
        if "taxi" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='taxi')
            transportation_metadata += description + ", "
            transportation_lst.append({"Description": description, "Information": info})

    return attractions_lst, accommodations_lst, restaurants_lst, transportation_lst, transportation_metadata

def get_reference_information_in_chunks_string(reference_information):
    """
    Splits the reference_information into attractions, accommodations, restaurants, and transportation. [STRINGS]
    """
    attractions_lst = []
    accommodations_lst = []
    restaurants_lst = []
    transportation_lst = []
    metadata = "Reference Information available about "
    for pair in reference_information:
        description = pair['Description']
        metadata += description + ", "
        if "attractions" in description.lower():
            city = description.split("Attractions in ")[1]
            info = attractions.run(city)
            attractions_lst.append({"Description": description, "Information": info.to_string()})

        if "accommodations" in description.lower():
            city = description.split("Accommodations in ")[1]
            info = accommodation.run(city)
            accommodations_lst.append({"Description": description, "Information": info.to_string()})

        if "restaurants" in description.lower():
            city = description.split("Restaurants in ")[1]
            info = restaurants.run(city)
            restaurants_lst.append({"Description": description, "Information": info.to_string()})

        if "flight" in description.lower():
            org_city, dest_city = extract_from_to_on(description)
            date = extract_date(description)
            info = flight.run(org_city, dest_city, date)
            transportation_lst.append({"Description": description, "Information": str(info)})

        if "driving" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='self-driving')
            
            transportation_lst.append({"Description": description, "Information": str(info)})
        
        if "taxi" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='taxi')
            transportation_lst.append({"Description": description, "Information": str(info)})

    return attractions_lst, accommodations_lst, restaurants_lst, transportation_lst, metadata

def get_reference_information_in_chunks_jsonified(reference_information):
    """
    Splits the reference_information into attractions, accommodations, restaurants, and transportation. [JSON LISTS]
    """
    attractions_lst = []
    accommodations_lst = []
    restaurants_lst = []
    transportation_lst = []
    metadata = "Reference Information available about "
    for pair in reference_information:
        description = pair['Description']
        metadata += description + ", "
        if "attractions" in description.lower():
            city = description.split("Attractions in ")[1]
            info = attractions.run(city)
            attractions_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})

        if "accommodations" in description.lower():
            city = description.split("Accommodations in ")[1]
            info = accommodation.run(city)
            accommodations_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})

        if "restaurants" in description.lower():
            city = description.split("Restaurants in ")[1]
            info = restaurants.run(city)
            restaurants_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})

        if "flight" in description.lower():
            org_city, dest_city = extract_from_to_on(description)
            date = extract_date(description)
            info = flight.run(org_city, dest_city, date)
            if isinstance(info, pd.DataFrame):
                transportation_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})
            elif isinstance(info, str):
                transportation_lst.append({"Description": description, "Information": info})

        if "driving" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='self-driving')
            # transportation_metadata += description + ", " # No need to add this to the metadata
            if isinstance(info, pd.DataFrame):
                transportation_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})
            elif isinstance(info, str):
                transportation_lst.append({"Description": description, "Information": info})
        
        if "taxi" in description.lower():
            org_city, dest_city = extract_from_to(description)
            info = googleDistanceMatrix.run(org_city, dest_city, mode='taxi')
            if isinstance(info, pd.DataFrame):
                transportation_lst.append({"Description": description, "Information": dataframe_to_list_of_dicts(info)})
            elif isinstance(info, str):
                transportation_lst.append({"Description": description, "Information": info})

    return attractions_lst, accommodations_lst, restaurants_lst, transportation_lst, metadata


def postprocess(json_formatter_reply):
    # Regex to match lists
    pattern = r'\[[^\[\]]*\]'

    # Find all matches
    matches = re.findall(pattern, json_formatter_reply)

    if matches:
        return json.loads(matches[0])
    else:
        return None
    
    
def create_agents(agents_descriptions, llm_config_list):
    agents = {}
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


def is_travelplanner_correct(detailed_evaluation):
    """
    Checks if the travel planner is correct based on the detailed evaluation.
    """
    for key, value in detailed_evaluation.items():
        if value == None:
            return False
        else:
            for k, v in value.items():
                if v[0] == False:
                    return False
    return True