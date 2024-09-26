from memory_profiler import memory_usage
import ijson

def process_json():
    with open("/home/weili3/VLSI-LLM/data_collection/netlist2.json", 'r', buffering=1024 * 1024) as file:
        parser = ijson.items(file, 'item')
        for i, item in enumerate(parser):
            print(item)
            break

mem_usage = memory_usage(process_json)
print(f"Memory usage: {mem_usage}")
