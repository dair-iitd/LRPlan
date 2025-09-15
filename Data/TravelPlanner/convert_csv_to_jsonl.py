import csv
import json

# Specify the input and output file names
csv_file = 'train.csv'
jsonl_file = 'train.jsonl'

# Open the CSV file for reading
with open(csv_file, mode='r', newline='', encoding='utf-8') as f_csv:
    # Read the CSV file using DictReader to convert each row into a dictionary
    csv_reader = csv.DictReader(f_csv)
    
    # Open the JSONL file for writing
    with open(jsonl_file, mode='w', encoding='utf-8') as f_jsonl:
        for row in csv_reader:
            # Convert the dictionary to a JSON string and write to the JSONL file
            f_jsonl.write(json.dumps(row) + '\n')

print(f"CSV file {csv_file} successfully converted to JSONL file {jsonl_file}.")