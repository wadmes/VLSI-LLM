"""
Fix the bug in data generation, that makes the type attribute becomes a type object, rather than its index

Moreover, it reads graphs and save some statistics:
"""
import glob
import pickle as pkl
import tqdm
import numpy as np
types = ["buf", "and", "or", "xor", "not", "nand", "nor", "xnor", "0", "1", "x", "input", "bb_input", "bb_output"]
types = {type:i for i, type in enumerate(types)}
count = np.array([0]*14)
node_size_list = []
edge_size_list = []
avg_degree_list = []
max_degree_list = []
degree_list = []
error_count = 0
# for graph in ./netlist_data/graph/*.pkl, set each node's type attribute to its index
pbar = tqdm.tqdm(glob.glob(f"./netlist_data/graph/*.pkl"))
for graph_path in pbar:
    try:
        with open(graph_path, 'rb') as f:
            G = pkl.load(f)
            for node in G.nodes(data=True):
                if "type" in str(node[1]['type']):
                    node[1]['type'] = types[str(node[1]['type']).split("'")[1]]
                count[node[1]['type']] += 1
            node_size_list.append(len(G.nodes))
            edge_size_list.append(len(G.edges))
            avg_degree_list.append(np.mean([degree for _, degree in G.degree()]))
            max_degree_list.append(np.max([degree for _, degree in G.degree()]))
            degree_list.append([degree for _, degree in G.degree()])
        
        with open(graph_path, 'wb') as f2:
            pkl.dump(G, f2)
    except Exception as e:
        print(e)
        error_count += 1
        continue
    # show error_count in tqdm bar
    pbar.set_description(f"error_count = {error_count}")
print("error_count = ", error_count)
print(count)
stat = {}
stat['node_size'] = node_size_list
stat['edge_size'] = edge_size_list
stat['avg_degree'] = avg_degree_list
stat['max_degree'] = max_degree_list
stat['degree'] = degree_list
stat['count'] = count
with open(f"./netlist_data/graph_stat.pkl", 'wb') as f:
    pkl.dump(stat, f)
