import json
import pickle as pkl
import csv
from tqdm import tqdm
import os

def get_file_length(file_path):
    with open(file_path, 'r') as file:
        return len(file.read())

def extract_graph_info(G):
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    out_degree_distribution = {}
    in_degree_distribution = {}
    type_distribution = {}
    output_distribution = {}
    for node, data in G.nodes(data=True):
        out_degree = G.out_degree(node)
        in_degree = G.in_degree(node)
        out_degree_distribution[out_degree] = out_degree_distribution.get(out_degree, 0) + 1
        in_degree_distribution[in_degree] = in_degree_distribution.get(in_degree, 0) + 1
        node_type = data['type']
        output = data['output']
        type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
        output_distribution[output] = output_distribution.get(output, 0) + 1
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "out_degree_distribution": out_degree_distribution,
        "in_degree_distribution": in_degree_distribution,
        "type_distribution": type_distribution,
        "output_distribution": output_distribution
    }

def create_csv_from_netlist_json(netlist_json, csv_file_path, netlist_data_dir):
    with open(netlist_json, "r") as file:
        data = json.load(file)
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = [
                "id", "rtl_id", "generic_effort", "mapping_effort", "optimization_effort",
                "#input", "#output", "#node", "#edge", "indegree_distribution", "outdegree_distribution", 
                "#not_node", "#nand_node", "#nor_node", "#xor_node", "#xnor_node", "#input_node", "#0_node",
                "#1_node", "#x_node", "#buf_node", "#and_node", "#or_node", "#bb_input_node", "#bb_output_node",
                "verilog_file_length"
            ]
            writer.writerow(header)
        for key, value in tqdm(data.items()):
            with open(f"{netlist_data_dir}/graph/{value['rtl_id']}_{value['synthesis_efforts']}.pkl", "rb") as graph_file:
                graph = pkl.load(graph_file)
                graph_info = extract_graph_info(graph)
                efforts = value['synthesis_efforts'].split('_') 
                row = [
                    int(key), value['rtl_id'], efforts[0], efforts[1], efforts[2],
                    graph_info["type_distribution"].get(11, 0),
                    graph_info["output_distribution"].get(1, 0),
                    graph_info["node_count"], graph_info["edge_count"], 
                    graph_info["in_degree_distribution"],
                    graph_info["out_degree_distribution"], 
                    graph_info["type_distribution"].get(4, 0),
                    graph_info["type_distribution"].get(5, 0),
                    graph_info["type_distribution"].get(6, 0),
                    graph_info["type_distribution"].get(3, 0),
                    graph_info["type_distribution"].get(7, 0),
                    graph_info["type_distribution"].get(11, 0),
                    graph_info["type_distribution"].get(8, 0),
                    graph_info["type_distribution"].get(9, 0),
                    graph_info["type_distribution"].get(10, 0),
                    graph_info["type_distribution"].get(0, 0),
                    graph_info["type_distribution"].get(1, 0),
                    graph_info["type_distribution"].get(2, 0),
                    graph_info["type_distribution"].get(12, 0),
                    graph_info["type_distribution"].get(13, 0),
                    get_file_length(f"{netlist_data_dir}/verilog/{value['rtl_id']}_{value['synthesis_efforts']}.v")
                ]
                writer.writerow(row)
