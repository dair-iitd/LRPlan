task = """
As an AI agent, your objective is to efficiently complete a series of tasks as described. You must adhere to the specific requirements and constraints of each task, including dependencies and timing. Efficiency is key; complete all tasks in the shortest possible time. I will provide instructions regarding actions and objects.\n**Action Protocol**:\n - You can perform only one action at a time-step/minute.\n - At each time-step, i.e., after EVERY MINUTE, output a valid action.\n - You need to figure out the dependencies amongst the actions and output the actions in correct order and timings for them to be valid.\n - Output the action explicitly and do not add other symbols (e.g., wash cup).\n - Actions fall into two types:\n    - Type 1: Action occupies you until completion (e.g., wash OBJ).\n    - Type 2: Action lets you be idle, allowing to perform other actions (e.g., heat OBJ).\n - Follow the \"Valid Actions\" for your output (e.g., wash cup).\n - If no action is required, use \"wait\" to skip the current time.\n - Output the full sequence of actions in a numbered sequence, starting from '0:' indicating the minute/time step the action is taken, seperated by newlines and enclosed with the tags <ANSWER></ANSWER>. While doing an action, you can output the action at the start of the minute/time-step and then wait for the next minute/time-step to output the next action if it as an idle action or takes 1 minute to complete, otherwise you have to do 'wait' actions in subsequent time-steps till the action is complete before outputting the next action.\n
"""

output_format = """
The output must be a numbered sequence of actions, starting from 0: for the first minute (time step 0) and incrementing the number by 1 for each subsequent minute. Each action should be written on a new line, specifying only the action (e.g., wash cup) without any additional text, explanations, or special symbols. If no valid action can be taken at a given minute, the agent must output wait. The entire sequence of actions must be enclosed between <ANSWER> and </ANSWER> tags. This strict format ensures the output is easy to parse and evaluate automatically.\n

SAMPLE OUTPUT:\n
<ANSWER>\n0: wash cup\n1: wait\n2: wash bedsheet\n3: wait\n4: wait\n5: dry cup\n...(and so on)</ANSWER>
"""

input_variables_descriptions = {
    "query": "The task description"
}

# IDX (cooking3), (household1, household2), 
input_output_pairs = [
    {
        "sample_input": {
            "query": """
            The maximum time allowed for completing all tasks is 40 minutes. Please cutoff you answer at the completion of all tasks or before this maximum limit. Note that this time limit is always higher than actually required time.Tasks:\n**Task Make tomato noodle stir-fry, which consists of cooked noodle and fried tomato.**\nValid Actions with time required for completion:\n- pick noodle: 1 minutes\n- cook noodle in pot: 5 minutes\n- add noodle to dish: 3 minutes\n- pick tomato: 2 minutes\n- chop tomato: 3 minutes\n- fry tomato in fryer: 2 minutes\n- add tomato to dish: 3 minutes\n- wash dish: 2 minutes\n
            """
        },
        "sample_output": """
        <ANSWER>\n0: wash dish\n1: wait\n2: pick noodle\n3: cook noodle in pot\n4: pick tomato\n5: wait\n6: chop tomato\n7: wait\n8: wait\n9: fry tomato in fryer\n10: add noodle to dish\n11: wait\n12: wait\n13: add tomato to dish\n14: wait\n15: wait\n</ANSWER>
        """
    },
    {
        "sample_input": {
            "query": """
            The maximum time allowed for completing all tasks is 80 minutes. Please cutoff you answer at the completion of all tasks or before this maximum limit. Note that this time limit is always higher than actually required time.Tasks:\n**Task Make a cup of tea.**\n**Task Clean the dishes using the dishwasher and dispose trash.**\nValid Actions with time required for completion:\n- activate kettle: 4 minutes\n- pour kettle into teapot: 2 minutes\n- wash teapot: 1 minutes\n- brew tea with teapot: 3 minutes\n- wash cup: 3 minutes\n- pour teapot into cup: 3 minutes\n- gather dishes: 3 minutes\n- scrape dishes into trash: 2 minutes\n- place dishes into dishwasher: 4 minutes\n- empty trash: 4 minutes\n- activate dishwasher: 4 minutes\n
            """
        },
        "sample_output": """
        <ANSWER>\n0: activate kettle\n1: wash teapot\n2: gather dishes\n3: wait\n4: wait\n5: pour kettle into teapot\n6: wait\n7: scrape dishes into trash\n8: wait\n9: brew tea with teapot\n10: wait\n11: wait\n12: empty trash\n13: wait\n14: wait\n15: wait\n16: wash cup\n17: wait\n18: wait\n19: place dishes into dishwasher\n20: wait\n21: wait\n22: wait\n23: activate dishwasher\n24: pour teapot into cup\n25: wait\n26: wait\n</ANSWER>
        """
    }
]