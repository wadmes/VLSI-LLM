import json
import pickle as pkl
import csv
from tqdm import tqdm

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

def create_csv_from_graph_data(json_data, csv_file_path):
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = [
            "id", "rtl_id", "generic_effort", "mapping_effort", "optimization_effort",
            "#input", "#output", "#node", "#edge", "indegree_distribution", "outdegree_distribution", 
            "#not_node", "#nand_node", "#nor_node", "#xor_node", "#xnor_node", "#input_node", "#0_node",
            "#1_node", "#x_node", "#buf_node", "#and_node", "#or_node", "#bb_input_node", "#bb_output_node"
        ]
        writer.writerow(header)
        i=0
        for key, value in tqdm(json_data.items()):
            with open(f"/home/weili3/VLSI-LLM/data_collection/MGVerilog11144/netlist_data/graph/{value['rtl_id']}_{value['synthesis_efforts']}.pkl", "rb") as graph_file:
                graph = pkl.load(graph_file)
                graph_info = extract_graph_info(graph)
                efforts = value['synthesis_efforts'].split('_') 
                #types = ["buf", "and", "or", "xor", "not", "nand", "nor", "xnor", "0", "1", "x", "input", "bb_input", "bb_output"]
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
                    graph_info["type_distribution"].get(13, 0)
                ]
                writer.writerow(row)


if __name__ == "__main__":
    with open("/home/weili3/VLSI-LLM/data_collection/MGVerilog11144/netlist_data/netlist.json", "r") as file:
        data = json.load(file)
    create_csv_from_graph_data(data, "/home/weili3/VLSI-LLM/data_collection/MGVerilog11144/netlist_data/netlist.csv")