import json
import pickle as pkl
import csv
from tqdm import tqdm
import os

def get_file_length(file_path):
    with open(file_path, 'r') as file:
        return len(file.read())

def extract_dataflow_graph_info(G):
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    out_degree_distribution = {}
    in_degree_distribution = {}
    type_distribution = {}
    for node, data in G.nodes(data=True):
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        out_degree_distribution[out_degree] = out_degree_distribution.get(out_degree, 0) + 1
        in_degree_distribution[in_degree] = in_degree_distribution.get(in_degree, 0) + 1
        if node.startswith("op_"):
            node_type = data['label']
        else:
            assert(node.startswith("signal_"))
            node_type = 'Signal'
        type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "out_degree_distribution": out_degree_distribution,
        "in_degree_distribution": in_degree_distribution,
        "type_distribution": type_distribution,
    }

def create_csv_from_rtl_data(netlist_json, csv_file_path, netlist_data_dir, synthesis_dir):
    with open(netlist_json, "r") as file:
        data = json.load(file)
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = [
                "rtl_id", "module_number", "module_name_list", "dataflow_status", "synthesis_status",
                "4o_label", "70b_label", "consistent_label", "verilog_file_length", "#dataflow_node", "#dataflow_edge",
                "dataflow_node_type_distribution", "dataflow_node_in_degree_distribution", "dataflow_node_out_degree_distribution"
            ]
            writer.writerow(header)
        for idx, v in tqdm(data.items()):
            rtl_id = int(idx)
            module_names = list(v['name_mapping'].keys())
            dataflow_status = all(os.path.isfile(f"{netlist_data_dir}/dataflow_graph/{rtl_id}_{module}.pkl") for module in module_names)
            verilog_file_length = get_file_length(f"{synthesis_dir}/{idx}/rtl.v")
            dataflow_status = 1 if dataflow_status else 0
            synthesis_status = 1 if v['synthesis_status'] else 0
            total_nodes = 0
            total_edges = 0
            total_type_distribution = {}
            total_in_degree_distribution = {}
            total_out_degree_distribution = {}
            for module in module_names:
                if os.path.isfile(f"{netlist_data_dir}/dataflow_graph/{rtl_id}_{module}.pkl"):
                    with open(f"{netlist_data_dir}/dataflow_graph/{rtl_id}_{module}.pkl", "rb") as graph_file:
                        graph = pkl.load(graph_file)
                        graph_info = extract_dataflow_graph_info(graph)
                        total_nodes += graph_info["node_count"]
                        total_edges += graph_info["edge_count"]
                        for key, value in graph_info["type_distribution"].items():
                            total_type_distribution[key] = total_type_distribution.get(key, 0) + value
                        for key, value in graph_info["in_degree_distribution"].items():
                            total_in_degree_distribution[key] = total_in_degree_distribution.get(key, 0) + value
                        for key, value in graph_info["out_degree_distribution"].items():
                            total_out_degree_distribution[key] = total_out_degree_distribution.get(key, 0) + value
            row = [
                rtl_id, len(module_names), module_names, dataflow_status, synthesis_status,
                v['4o_label'], v['70b_label'], v['consistent_label'], verilog_file_length,
                total_nodes, total_edges, total_type_distribution, total_in_degree_distribution, total_out_degree_distribution
            ]
            writer.writerow(row)
