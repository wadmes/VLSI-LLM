#!/usr/bin/python3
import json
import re
import os

'''
This script is used to parse the rtl_data.json file (at /home/weili3/VLSI-LLM/data_collection/rtl_data/rtl_with_label.json)
and generate a new json file that contains the names of all verilog files for each rtl item, 
as well as the largest and smallest verilog files for each rtl item.
It also generates a txt file with all the unlabeled verilog files sorted by size.

Input:
    - rtl_data_with_label.json (Williams output): the json file that contains the rtl data with function labels
Output:
    - labeled_items_with_size.json: the json file that contains all labeled rtl items successfully synthesized, including the names of all verilog files for each rtl item, 
    as well as the largest and smallest verilog files for each rtl item.
    - unlabeled_items_with_size.json: the json file that contains all unlabeled rtl items successfully synthesized, including the names of all verilog files for each rtl item, 
    as well as the largest and smallest verilog files for each rtl item.
    - unsynthesisized_items.json: the json file that contains all unsynthesized rtl items. 
'''


def parse_items(input_file):
    try:
        # Read the input JSON file
        with open(input_file, 'r') as f:
            data = json.load(f)
            #print("length of data: ", len(data.keys()))
        
        # Filter items with non-empty function_label
        labeled_data = {
            k: v for k, v in data.items() 
            if v.get('function_label') and v['function_label'].strip()
        }
        
        # Get unlabeled items
        unlabeled_data = {
            k: v for k, v in data.items() 
            if not v.get('function_label') or not v['function_label'].strip()
        }
        
        # Save filtered data to new file
        #with open("labeled_items.json", 'w') as f:
        #    json.dump(labeled_data, f, indent=4)
            
        # Save unlabeled data to new file
        #with open("unlabeled_items.json", 'w') as f:
        #    json.dump(unlabeled_data, f, indent=4)
            
        #print(f"Successfully saved {len(filtered_data)} labeled items to {output_file}")
        #print(f"Successfully saved {len(unlabeled_data)} unlabeled items to {unlabeled_file}")
        return labeled_data, unlabeled_data
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_file}")
    except Exception as e:
        print(f"Error: {str(e)}")

def parse_file_listing(input_file):
    import json
    import re
    
    file_data = {}
    index = 0
    # Read input file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse each line
    for line in lines:
        # Skip total line and empty lines
        if line.startswith('total') or not line.strip():
            continue
            
        # Split line into components
        parts = line.split()
        if len(parts) < 5:
            continue
            
        size = parts[4]  # File size is 5th column
        name = parts[-1] # Filename is last column
        
        # Only include .v files
        if name.endswith('.v'):
            file_data[index] = {
                'name': name,
                'size': int(size)
            }
            index += 1
    
    # Write to JSON file
    return file_data


def merge_files(file_sizes, labeled_items):
    # Load both JSON files

    
    for item in labeled_items:
        labeled_items[item]["verilog_files"] = {}
    
    # Track unmatched items
    unmatched = {}
    
    # Process each file size entry
    for idx, item in file_sizes.items():
        # Extract the numerical ID from the name field
        match = re.match(r'(\d+)_', item['name'])
        if not match:
            unmatched[idx] = item
            continue
            
        file_id = match.group(1)
        
        # Check if ID exists in labeled_items
        if file_id in labeled_items:
            # Append all key-value pairs from file_sizes to labeled_items
            labeled_items[file_id]["verilog_files"][idx] = item
        else:
            unmatched[idx] = item
    
    return labeled_items, unmatched
    

def get_unlabeled_names(data, output_path = "unlabeled_files.txt"):

    unlabeled_data = {}
    for item in data.keys():
        for verilog in data[item]["verilog_files"].keys():
            unlabeled_data[data[item]["verilog_files"][verilog]["name"]] = data[item]["verilog_files"][verilog]["size"]
    sorted_by_values = sorted(unlabeled_data.items(), key=lambda item: item[1])
    
    with open(output_path, 'w') as f:
        for name, size in sorted_by_values:
            f.write(f"{name}\n")

    with open("unlabeled_files_with_size.txt", 'w') as f:
        for name, size in sorted_by_values:
            f.write(f"{name}: {size}\n")



def get_max_min_size(items):

    
        
    for item in items.keys():
        try:
            # Get all files for this item
            files = items[item]["verilog_files"].keys()
            # Find max and min by comparing 'size' field
            items[item]["largest_verilog_file"] = {}
            items[item]["smallest_verilog_file"] = {}
            max_size = max(files, key=lambda x: items[item]["verilog_files"][x]['size'])
            max_value = items[item]["verilog_files"][max_size]['size']
            min_size = min(files, key=lambda x: items[item]["verilog_files"][x]['size'])
            min_value = items[item]["verilog_files"][min_size]['size']
            items[item]["largest_verilog_file"][max_size] = max_value
            items[item]["smallest_verilog_file"][min_size] = min_value

        except:
            continue

            
    return items


def write_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def seperate_synthesisized_items(data):
    anomoly_items = {}
    normal_synthesisized_data = {}
    for item in data.keys():
        if len(data[item]["verilog_files"]) == 0:
            anomoly_items[item] = data[item]
        else:
            normal_synthesisized_data[item] = data[item]
    return anomoly_items, normal_synthesisized_data

#get labeled and unlabeled items
labeled_data, unlabeled_data = parse_items("/home/weili3/VLSI-LLM/data_collection/rtl_data/rtl_with_label.json")


# get files by size (small to large)
os.system("ls -lSr /home/weili3/VLSI-LLM/data_collection/netlist_data/verilog/ > out_long.txt")
file_sizes =parse_file_listing('out_long.txt')

#merge together file_size and labeled_items jsons
labeled_items, unmatched = merge_files(file_sizes, labeled_data)

#merge together unmatched and unlabeled_items jsons
unlabeled_items, unmatched2 = merge_files(unmatched, unlabeled_data)

#seperate synthesisized and unsynthesisized items from unlabeled data
unlabeled_unsynthesisized_items, unlabeled_synthesisized_items = seperate_synthesisized_items(unlabeled_items)
labeled_unsynthesisized_items, labeled_synthesisized_items = seperate_synthesisized_items(labeled_items)

#seperate synthesisized and unsynthesisized items from unlabeled data
#synthesisized_items, unsynthesisized_items = seperate_synthesisized_items(unlabeled_items)


print("----RTL Statistics----")
print(f"Number of labeled RTL items: {len(labeled_items.keys())}")
print(f"Number of unlabeled RTL items: {len(unlabeled_items.keys())}")
print(f"Total number of RTL items: {len(labeled_items.keys()) + len(unlabeled_items.keys())}")

print(f"\n----Synthesisized RTL Statistics----")


#get all unlabeled synthesisized verilog file names for future processing
#get_unlabeled_names(unlabeled_synthesisized_items, "unlabeled_files.txt")

#print(f"Number of unlabeled RTL items without synthesisized verilog files: {len(unlabeled_unsynthesisized_items.keys())}")
#print(f"Number of unlabeled RTL items with synthesisized verilog files: {len(unlabeled_synthesisized_items.keys())}")
#print(f"Number of labeled RTL items without synthesisized verilog files: {len(labeled_unsynthesisized_items.keys())}")
#print(f"Number of labeled RTL items with synthesisized verilog files: {len(labeled_synthesisized_items.keys())}")

print(f"\nTotal number of RTL items: {len(unlabeled_unsynthesisized_items.keys()) + len(unlabeled_synthesisized_items.keys()) + len(labeled_unsynthesisized_items.keys()) + len(labeled_synthesisized_items.keys())}")

print(f"\nTotal number of RTL items with synthesisized verilog files: {len(unlabeled_synthesisized_items.keys()) + len(labeled_synthesisized_items.keys())}")
print(f"Unlabeled: {len(unlabeled_synthesisized_items.keys())}")
print(f"Labeled: {len(labeled_synthesisized_items.keys())}")

print(f"\nTotal number of RTL items without synthesisized verilog files: {len(unlabeled_unsynthesisized_items.keys()) + len(labeled_unsynthesisized_items.keys())}")
print(f"Unlabeled: {len(unlabeled_unsynthesisized_items.keys())}")
print(f"Labeled: {len(labeled_unsynthesisized_items.keys())}")


#get max and min sizes from data
min_max_labeled = get_max_min_size(labeled_synthesisized_items)
min_max_unlabeled = get_max_min_size(unlabeled_synthesisized_items)

write_json(min_max_labeled, 'labeled_items_with_size.json')
write_json(min_max_unlabeled, 'unlabeled_items_with_size.json')
labeled_unsynthesisized_items.update(unlabeled_unsynthesisized_items)
write_json(labeled_unsynthesisized_items, 'unsynthesisized_items.json')

print(f"\nwrote labeled items to labeled_items_with_size.json with {len(min_max_labeled.keys())} items")
print(f"wrote unlabeled items to unlabeled_items_with_size.json with {len(min_max_unlabeled.keys())} items")
print(f"wrote unsynthesisized items to unsynthesisized_items.json with {len(labeled_unsynthesisized_items.keys())} items")
