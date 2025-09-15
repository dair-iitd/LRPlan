# LRPlan

## Setup
1. Create a conda environment and install dependencies:
```bash
conda create -n lrplan python=3.9
conda activate lrplan
pip install -r requirements.txt
```
2. (Only needed for TravelPlanner) Download the [database](https://drive.google.com/file/d/1pF1Sw6pBmq2sFkJvm-LzJOqrmfWoQgxE/view?usp=drive_link) and unzip it to the `Code/TravelPlanner/feedback_script/travelplanner/` directory (creating a folder named 'database' inside `Code/TravelPlanner/feedback_script/travelplanner/`).

## Running

### TravelPlanner
1. Write API Keys in `OAI_CONFIG_LIST`
2. Navigate to the directory `Code/TravelPlanner/`
3. Generate reasoning traces:
    1. Write your DeepSeek API key at `line 33` in the file `get_reasoning_traces.py`
    2. Navigate back to the main directory.
    ```bash
    python get_reasoning_traces.py
    ```
4. Run Inference file in directory `Code/TravelPlanner/LRPlan`:
```bash
python LRPlan.py 
```
### TimeArena-Static
1. Similar to TravelPlanner in the `Code/TimeArenaStatic/` directory.

## Prompts
1. All prompts are found in the directory `Prompts/`.
2. `cot_instructions.py` contains basic CoT instructions used for baseline comparisons.
3. `pattern_extractor_corrector.py` contains the prompt for meta-agents used in LRPlan.
4. `timearena.py` and `travelplanner` contain task description, output format, and a few sample input-output pairs for their respective datasets.

## Data
1. Directory `Data/` contains all query samples for both datasets.
