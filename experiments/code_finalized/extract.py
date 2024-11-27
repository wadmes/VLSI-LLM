import json
import logging
import argparse
from typing import List, Dict
from multiprocessing import Pool
import os
from tqdm import tqdm

def read_file(verilog_dir, file):
    """
    Reads and preprocesses a Verilog file from a specified directory.

    Args:
        verilog_dir (str): Directory containing Verilog files
        file (tuple): A tuple containing (index, filename) for the Verilog file

    Returns:
        tuple: A tuple containing (index, dict) where dict contains either:
            - {'verilog': preprocessed_content, 'file_name': filename} for successful reads
            - {'error': error_message, 'file_name': filename} for failed reads
    """
    idx, filename = file
    try:
        with open(os.path.join(verilog_dir, filename), 'r') as f:
            raw_content = f.read()
            # Preprocess to remove comments and extra whitespace
            clean_content = preprocess_verilog(raw_content)
            #clean_content = raw_content
            return idx, {'verilog': clean_content, 'file_name': filename}
    except Exception as e:
        return idx, {'error': str(e), 'file_name': filename}

def preprocess_verilog(content: str) -> str:
    """
    Preprocesses Verilog content by removing comments and extra whitespace.

    Args:
        content (str): Raw Verilog file content

    Returns:
        str: Cleaned Verilog content with comments and extra whitespace removed
    """
    import re
    # Remove single-line comments
    content = re.sub(r'//.*', '', content)
    # Remove multi-line comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    # Remove extra whitespace
    content = ' '.join(content.split())

    return content

def count_function_labels(json_file):
    """
    Analyzes a JSON file to count and calculate statistics for function labels.

    Args:
        json_file (str): Path to the JSON file containing function labels
    Prints:
        - Individual label counts and their percentages
        - Total count of all labels
    """
    counts = {}
    print(f"Counting for: {json_file}")
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        
       
        # Iterate through the data
        for item in data.keys():
            label = data[item]['function_label']
            counts[label] = counts.get(label, 0) + 1
            
                    
        # Sort by count in descending order
        counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        
        print("\nFunction Label Counts:")
        total=0
        for label, count in counts.items():
            print(f"{label}: {count}, Percentage: {(count/len(data.keys()))*100:.2f}%")
            total += count
        print(f"Total: {total}")

    except FileNotFoundError:
        logging.error(f"File {json_file} not found")
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")


def count_mg_mismatches(json_file_path):
    # Fields to compare against target (excluding rtl_id and synthesis_efforts)
    prediction_fields = [
    "3.1-8b-2k-label",
    "3.1-8b-2k-label-rtl",
    "3.1-8b-2k-description-rtl",
    "3.2-1B-2k-label-rtl",
    "3.2-3B-2k-label-rtl",
    "3.1-8B-4k-label-rtl",
    "3.2-3B-2k-label",
    "3.1-8B-4k-label"
    ]
    
    mismatches = {}
    label_stats = {}  # Nested dict to store per-label statistics
    total_entries = 0
    
    with open(json_file_path) as f:
        data = json.load(f)
        
    # Initialize statistics tracking
    for pred_field in prediction_fields:
        mismatches[pred_field] = 0
        label_stats[pred_field] = {
            'matches': {},      # Correct predictions by label
            'mismatches': {},   # Incorrect predictions by label
            'confusion': {}     # Predicted vs actual for mismatches
        }
    
    for entry_id, entry in data.items():
        if "target_label" not in entry or not entry["target_label"]:
            continue
            
        total_entries += 1
        target = entry["target_label"]
        
        # Compare each prediction field against target
        for pred_field in prediction_fields:
            prediction = entry[pred_field]
            
            # Initialize counters if needed
            if target not in label_stats[pred_field]['matches']:
                label_stats[pred_field]['matches'][target] = 0
            if target not in label_stats[pred_field]['mismatches']:
                label_stats[pred_field]['mismatches'][target] = 0
            
            if prediction != target:
                mismatches[pred_field] += 1
                label_stats[pred_field]['mismatches'][target] += 1
                
                # Track confusion matrix data
                confusion_key = f"{target}->{prediction}"
                label_stats[pred_field]['confusion'][confusion_key] = \
                    label_stats[pred_field]['confusion'].get(confusion_key, 0) + 1
            else:
                label_stats[pred_field]['matches'][target] += 1
    
    # Print results
    print(f"\nTotal entries analyzed: {total_entries}")
    print(f"\nTotal entries not analyzed due to missing consistent labels: {17895 - total_entries}")
    print("\nOverall Statistics:")
    print("=" * 50)
    for field, count in mismatches.items():
        accuracy = ((total_entries - count) / total_entries) * 100
        print(f"{field}: {count} mismatches ({accuracy:.2f}% accuracy)")
    
    # Print detailed statistics for each field
    for field in prediction_fields:
        print(f"\nDetailed Statistics for {field}")
        print("=" * 50)
        
        print("\nAccuracy by Label:")
        print("-" * 30)
        for label in set(label_stats[field]['matches'].keys()) | set(label_stats[field]['mismatches'].keys()):
            matches = label_stats[field]['matches'].get(label, 0)
            mismatches = label_stats[field]['mismatches'].get(label, 0)
            total = matches + mismatches
            accuracy = (matches / total * 100) if total > 0 else 0
            print(f"{label}:")
            print(f"  Correct: {matches}")
            print(f"  Incorrect: {mismatches}")
            print(f"  Accuracy: {accuracy:.2f}%")
        
        print("\nMost Common Misclassifications:")
        print("-" * 30)
        sorted_confusion = sorted(
            label_stats[field]['confusion'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for (confusion_pair, count) in sorted_confusion[:5]:  # Show top 5 misclassifications
            true_label, pred_label = confusion_pair.split("->")
            print(f"True: {true_label} -> Predicted: {pred_label}: {count} times")
        
        print("\n")

def count_rtl_mismatches(json_file_path):
    # Fields to compare against target (excluding rtl_id and synthesis_efforts)
    prediction_fields = [
        "3.2-1B-2k-prediction",
        "3.1-8B-2k-prediction", 
        "3.2-1B-4k-prediction",
        "3.2-3B-2k-prediction",
        "3.2-3B-4k-prediction",
        "3.1-8B-4k-prediction",
        "3.1-8B-2k-prediction-rtl",
        "3.2-3B-2k-prediction-rtl",
        "3.1-8B-4k-prediction-rtl"

    ]
    
    mismatches = {}
    label_stats = {}  # Nested dict to store per-label statistics
    total_entries = 0
    
    with open(json_file_path) as f:
        data = json.load(f)
        
    # Initialize statistics tracking
    for pred_field in prediction_fields:
        mismatches[pred_field] = 0
        label_stats[pred_field] = {
            'matches': {},      # Correct predictions by label
            'mismatches': {},   # Incorrect predictions by label
            'confusion': {}     # Predicted vs actual for mismatches
        }
    
    for entry_id, entry in data.items():
        if "target" not in entry or not entry["target"]:
            continue
            
        total_entries += 1
        target = entry["target"]
        
        # Compare each prediction field against target
        for pred_field in prediction_fields:
            prediction = entry[pred_field]
            
            # Initialize counters if needed
            if target not in label_stats[pred_field]['matches']:
                label_stats[pred_field]['matches'][target] = 0
            if target not in label_stats[pred_field]['mismatches']:
                label_stats[pred_field]['mismatches'][target] = 0
            
            if prediction != target:
                mismatches[pred_field] += 1
                label_stats[pred_field]['mismatches'][target] += 1
                
                # Track confusion matrix data
                confusion_key = f"{target}->{prediction}"
                label_stats[pred_field]['confusion'][confusion_key] = \
                    label_stats[pred_field]['confusion'].get(confusion_key, 0) + 1
            else:
                label_stats[pred_field]['matches'][target] += 1
    
    # Print results
    print(f"\nTotal entries analyzed: {total_entries}")
    print(f"\nTotal entries not analyzed due to missing consistent labels: {17895 - total_entries}")
    print("\nOverall Statistics:")
    print("=" * 50)
    for field, count in mismatches.items():
        accuracy = ((total_entries - count) / total_entries) * 100
        print(f"{field}: {count} mismatches ({accuracy:.2f}% accuracy)")
    
    # Print detailed statistics for each field
    for field in prediction_fields:
        print(f"\nDetailed Statistics for {field}")
        print("=" * 50)
        
        print("\nAccuracy by Label:")
        print("-" * 30)
        for label in set(label_stats[field]['matches'].keys()) | set(label_stats[field]['mismatches'].keys()):
            matches = label_stats[field]['matches'].get(label, 0)
            mismatches = label_stats[field]['mismatches'].get(label, 0)
            total = matches + mismatches
            accuracy = (matches / total * 100) if total > 0 else 0
            print(f"{label}:")
            print(f"  Correct: {matches}")
            print(f"  Incorrect: {mismatches}")
            print(f"  Accuracy: {accuracy:.2f}%")
        
        print("\nMost Common Misclassifications:")
        print("-" * 30)
        sorted_confusion = sorted(
            label_stats[field]['confusion'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for (confusion_pair, count) in sorted_confusion[:5]:  # Show top 5 misclassifications
            true_label, pred_label = confusion_pair.split("->")
            print(f"True: {true_label} -> Predicted: {pred_label}: {count} times")
        
        print("\n")

def merge_label_predictions(input_file, file_to_merge, new_field):
    with open(input_file, 'r') as file:
        data = json.load(file)
    with open(file_to_merge, 'r') as file:
        data2 = json.load(file)
    
    # Get ordered lists of items from both JSONs
    data_items = list(data.items())
    data2_items = list(data2.items())
    
    converted_dict = {}
    for i in range(len(data_items)):
        first_key, first_data = data_items[i]
        second_key, second_data = data2_items[i]
        data[first_key][new_field] = second_data["function_label"]
        
    print(len(data.keys()))
    with open(input_file, 'w') as file:
        json.dump(data, file, indent=2)

def merge_label_predictions(input_file, file_to_merge, new_field):
    with open(input_file, 'r') as file:
        data = json.load(file)
    with open(file_to_merge, 'r') as file:
        data2 = json.load(file)
    
    # Get ordered lists of items from both JSONs
    data_items = list(data.items())
    data2_items = list(data2.items())
    
    converted_dict = {}
    for i in range(len(data_items)):
        first_key, first_data = data_items[i]
        second_key, second_data = data2_items[i]
        data[first_key][new_field] = second_data["description"]
        
    print(len(data.keys()))
    with open(input_file, 'w') as file:
        json.dump(data, file, indent=2)

def get_rtl_descriptions(input_json, rtl_json, output_json):
    with open(input_json, 'r') as file:
        data = json.load(file)
    with open(rtl_json, 'r') as file:
        rtl = json.load(file)

    for i in data.keys():
        rtl_id = data[i]['rtl_id']
        data[i]['target'] = rtl[str(rtl_id)]['description']
    with open(output_json, 'w') as file:
        json.dump(data, file, indent=2)

def generate_binary_matches(json_data, output_dir="match_results"):
    import os
    import csv
    from collections import defaultdict

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the prediction columns we want to compare against target
    prediction_cols = [
        "3.1-8B-2k-prediction-rtl",
        "3.1-8B-4k-prediction-rtl",
        "3.2-3B-2k-prediction-rtl"
    ]
    
    # Initialize results dictionary for each prediction column
    results = {col: defaultdict(list) for col in prediction_cols}
    
    # Process each entry
    for key, entry in json_data.items():
        # Skip if no target
        if "target" not in entry or not entry["target"]:
            continue
            
        target = entry["target"]
        
        # For each prediction column, compare with target
        for pred_col in prediction_cols:
            if pred_col in entry:
                # 1 for match, 0 for mismatch
                match = 1 if entry[pred_col] == target else 0
                results[pred_col]["key"].append(key)
                results[pred_col]["match"].append(match)
    
    # Write results to CSV files
    for pred_col in prediction_cols:
        filename = os.path.join(output_dir, f"{pred_col}_matches.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["key", "match"])  # Header
            for i in range(len(results[pred_col]["key"])):
                writer.writerow([
                    results[pred_col]["key"][i],
                    results[pred_col]["match"][i]
                ])

    return results


def generate_mg_binary_matches(json_data, output_dir="mg_results"):
    import os
    import csv
    from collections import defaultdict

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the prediction columns we want to compare against target
    prediction_cols = [
        "3.1-8b-2k-label",
    "3.1-8b-2k-label-rtl",
    "3.1-8b-2k-description-rtl",
    "3.2-1B-2k-label-rtl",
    "3.2-3B-2k-label-rtl",
    "3.1-8B-4k-label-rtl",
    "3.2-3B-2k-label",
    "3.1-8B-4k-label"
    ]
    
    # Initialize results dictionary for each prediction column
    results = {col: defaultdict(list) for col in prediction_cols}
    
    # Process each entry
    for key, entry in json_data.items():
        # Skip if no target
        if "target_label" not in entry or not entry["target_label"]:
            continue
            
        target = entry["target_label"]
        
        # For each prediction column, compare with target
        for pred_col in prediction_cols:
            if pred_col in entry:
                # 1 for match, 0 for mismatch
                match = 1 if entry[pred_col] == target else 0
                results[pred_col]["key"].append(key)
                results[pred_col]["match"].append(match)
    
    # Write results to CSV files
    for pred_col in prediction_cols:
        filename = os.path.join(output_dir, f"{pred_col}_matches.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["key", "match"])  # Header
            for i in range(len(results[pred_col]["key"])):
                writer.writerow([
                    results[pred_col]["key"][i],
                    results[pred_col]["match"][i]
                ])

    return results

def main():
    pass

if __name__ == "__main__":
    main()