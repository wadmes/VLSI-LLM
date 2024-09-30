import ijson
import json

if __name__ == "__main__":
    with open("/home/weili3/VLSI-LLM/data_collection/netlist2.json", 'r') as file:
        parser = ijson.items(file, 'item')
        for i, item in enumerate(parser):
            print(item['rtl_id'])

    # # Load the JSON file into a Python dictionary
    # with open("/home/weili3/VLSI-LLM/data_collection/RTL.json", 'r') as file:
    #     data = json.load(file)

    # # Print the number of key-value pairs in the dictionary
    # print(f"The length of the JSON dictionary is: {len(data)}")


