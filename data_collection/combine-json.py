# This script adds one key-value in one json file to another json file
import json
import argparse


"""
Load values from json1 and add them to json2
"""
def combine_json(json1, json2, key, new_key):
    with open(json1, 'r') as f:
        data1 = json.load(f)
    with open(json2, 'r') as f:
        data2 = json.load(f)
    for k, v in data1.items():
        data2[k][new_key] = v[key]
    with open(json2, 'w') as f:
        json.dump(data2, f,indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json1',type=str, default="./backupDataLog/rtl_data/RTL_with_desc.json", help='json file to be added')
    parser.add_argument('--json2',type=str, default="./rtl_data/rtl.json", help='json file to be added to')
    parser.add_argument('--key', type=str, default="rtl_description", help='key to be extracted from json1')
    # new_key = "rtl_description"
    parser.add_argument('--new_key', type=str, default="description", help='key to be added in json2')
    args = parser.parse_args()
    combine_json(args.json1, args.json2, args.key, args.new_key)