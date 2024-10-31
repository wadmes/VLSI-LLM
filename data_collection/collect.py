import os
import json
import re
import shutil
import networkx as nx
import circuitgraph as cg # type: ignore
import multiprocessing as mp
import pickle as pkl
from tqdm import tqdm
from itertools import product
LLM_DATA_DIR = "/storage/yangzou/gf/llm_data"
BBOXES = [cg.io.BlackBox("f", ["CK", "D"], ["Q"])] + cg.io.genus_flops + cg.io.dc_flops

# generate rtl_json
def generate_rtl_json():
    # helper func used to anonymize the module name
    def anonymize_modules(verilog_code):
        module_mapping = {}
        counter = 0
        pattern = r'(^module\s+)(\w+)(\s*\()'
        def replace_module_name(match):
            nonlocal counter
            original_module_name = match.group(2)
            anonymized_module_name = f"anonymized_module_{counter}"
            module_mapping[anonymized_module_name] = original_module_name
            counter += 1
            return f"{match.group(1)}{anonymized_module_name}{match.group(3)}"
        anonymized_code = re.sub(pattern, replace_module_name, verilog_code, flags=re.MULTILINE)
        return anonymized_code, module_mapping

    # retrieve rtl_ids that synthesized or produced dataflow successfully
    with open(f"{LLM_DATA_DIR}/synthesis_summary.pkl", 'rb') as f:
        success_synthesis, _, _, _, _ = pkl.load(f)
    with open(f"{LLM_DATA_DIR}/Pyverilog_analysis/success_dataflow.pkl", 'rb') as f:
        success_dataflow = pkl.load(f)
    success_dataflow = list(set([idx for idx, _ in success_dataflow]))
    
    # write json
    with open(f"{LLM_DATA_DIR}/rtl_data/rtl.json", mode='w') as file:
        file.write("{\n")
    is_first_entry = True
    for rtl_id in tqdm(range(26532)):
        rtl_path = f"{LLM_DATA_DIR}/bms/{rtl_id}"
        with open(f"{rtl_path}/rtl.sv", 'r') as file:
            verilog = file.read()
        anonymized_verilog, module_mapping = anonymize_modules(verilog)
        with open(f"{rtl_path}/instruction.txt", 'r') as file:
            instruction = file.read()
        record = {
            'verilog': anonymized_verilog,
            'module_mapping': module_mapping,
            'instruction': instruction,
            'description': '',
            'synthesis_status': rtl_id in success_synthesis,
            'dataflow_status': rtl_id in success_dataflow,
        }
        with open(f"{LLM_DATA_DIR}/rtl_data/rtl.json", mode='a') as file:
            if not is_first_entry:
                file.write(",\n")
            file.write(f'"{rtl_id}": ')
            json.dump(record, file, indent=4)
            is_first_entry = False
    with open(f"{LLM_DATA_DIR}/rtl_data/rtl.json", mode='a') as file:
        file.write("\n}")

# netlist graph generation function using cg package
def generate_netlist_graph(rtl_id_i):
    # helper func used to remove "_module" in module name to avoid graph generation error
    def remove_module_suffix(verilog_file, output_file):
        with open(verilog_file, 'r') as file:
            verilog_code = file.read()
        updated_verilog_code = verilog_code.replace('_module', '')
        with open(output_file, 'w') as output:
            output.write(updated_verilog_code)

    # helper func used to anonymize the node.name, node.data["type"], and node.data["output"]
    def anonymize_graph(G):
        types = ["buf", "and", "or", "xor", "not", "nand", "nor", "xnor", "0", "1", "x", "input", "bb_input", "bb_output"]
        types = {type:i for i, type in enumerate(types)}
        for name, data in G.nodes(data=True):
            data['type'] = types[data['type']]
            data['output'] = int(data['output'])
            data['name'] = name # save original name as a new name attribute
        return nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})

    # retrieve data and define path
    rtl_id, i, efforts = rtl_id_i
    syn_path = f"{LLM_DATA_DIR}/bms/{rtl_id}/syn/{i}"
    
    # try generate graph and save as pkl file
    try:
        remove_module_suffix(f"{syn_path}/syn.v", f"{syn_path}/syn_copy.v")
        c = cg.from_file(f"{syn_path}/syn_copy.v", blackboxes=BBOXES)
        G = anonymize_graph(c.graph)
        with open(f"{LLM_DATA_DIR}/netlist/graph/{rtl_id}_{efforts[i]}.pkl", 'wb') as file:
            pkl.dump(G, file)
        os.remove(f"{syn_path}/syn_copy.v")
    except:
        if os.path.exists(f"{syn_path}/syn_copy.v"):
            os.remove(f"{syn_path}/syn_copy.v")
        return (rtl_id, i)
    
    # return None if successful
    return None

# multithreads for netlist graph generation
def generate_netlist_graph_multithreads(num_cores):
    # parallel the generation
    with open(f"{LLM_DATA_DIR}/synthesis_summary.pkl", 'rb') as f:
        success, _, _, _, _ = pkl.load(f)
    fail = []
    efforts = [f'{ef1}_{ef2}_{ef3}' for ef1, ef2, ef3 in product(("low", "medium", "high"), repeat=3)]  
    inputs = [(rtl_id, i, efforts) for rtl_id in success for i in range(27)]
    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(generate_netlist_graph, inputs), total=len(inputs)))

    # retrieve and save failed rtl_id and i (synthesis effors) from non-None return value
    fail = [res for res in results if res is not None]
    with open(f"{LLM_DATA_DIR}/netlist/fail_netlist_graph.pkl", 'wb') as f:
        pkl.dump(fail, f)

# generate netlist json and save corresponding netlist and synthesis log according to the success/failure status 
def generate_netlist_json():
    with open(f"{LLM_DATA_DIR}/synthesis_summary.pkl", 'rb') as f:
        success, _, _, _, _ = pkl.load(f)
    with open(f"{LLM_DATA_DIR}/netlist/fail_netlist_graph.pkl", 'rb') as f:
        fail = pkl.load(f)
    idx = 0
    records_dict = {}
    efforts = [f'{ef1}_{ef2}_{ef3}' for ef1, ef2, ef3 in product(("low", "medium", "high"), repeat=3)]
    for rtl_id, i in tqdm([(rtl_id, i) for rtl_id in success for i in range(27)]):
        if (rtl_id, i) not in fail:
            syn_path = f"{LLM_DATA_DIR}/bms/{rtl_id}/syn/{i}"
            shutil.copy(f"{syn_path}/syn.v", f"{LLM_DATA_DIR}/netlist/verilog/{rtl_id}_{efforts[i]}.v")
            shutil.copy(f"{syn_path}/genus.log", f"{LLM_DATA_DIR}/netlist/synthesis_log/{rtl_id}_{efforts[i]}.log")
            records_dict[idx] = {
                'rtl_id': rtl_id,
                'synthesis_efforts': efforts[i],
            }
            idx += 1
    with open(f"{LLM_DATA_DIR}/netlist/netlist.json", mode='w') as file:
        json.dump(records_dict, file, indent=4)


generate_netlist_graph_multithreads(16)