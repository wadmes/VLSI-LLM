import os
import networkx as nx
import circuitgraph as cg
import multiprocessing as mp
import pickle as pkl
import typer
from tqdm import tqdm
from pathlib import Path
from itertools import product
from typing import Tuple, Optional

def process_netlist(args: Tuple[int, str, Path]) -> Optional[Tuple[int, str]]:
    """
    Process a single netlist file: clean up module names, anonymize its graph representation,
    and save the processed graph as a pickle file.

    Args:
        args (Tuple[int, str, Path]): 
            - rtl_id (int): ID of the RTL file.
            - effort (str): Synthesis effort level combination.
            - data_dir (Path): Base directory containing synthesis and output folders.

    Returns:
        Optional[Tuple[int, str]]: Returns (rtl_id, effort) if processing fails, None otherwise.
    """

    def remove_module_suffix(verilog_file: Path, output_file: Path) -> None:
        """
        Removes '_module' suffix from module names in a Verilog file.

        Args:
            verilog_file (Path): Input Verilog file path.
            output_file (Path): Output Verilog file path.
        """
        with open(verilog_file, 'r') as file:
            verilog_code = file.read()
        updated_verilog_code = verilog_code.replace('_module', '')
        with open(output_file, 'w') as output:
            output.write(updated_verilog_code)

    def anonymize_graph(G: nx.DiGraph) -> nx.DiGraph:
        """
        Anonymizes graph nodes by replacing their 'type' and relabeling the nodes.

        Args:
            G (nx.DiGraph): The circuit graph.

        Returns:
            nx.DiGraph: The anonymized graph.
        """
        types = ["buf", "and", "or", "xor", "not", "nand", "nor", "xnor", "0", "1", "x", "input", "bb_input", "bb_output"]
        type_mapping = {type_: i for i, type_ in enumerate(types)}

        for name, data in G.nodes(data=True):
            data['type'] = type_mapping[data['type']]
            data['output'] = int(data['output'])
            data['name'] = name
        return nx.relabel_nodes(G, {node: i for i, node in enumerate(G.nodes())})

    rtl_id, effort, data_dir = args
    syn_path = data_dir / f"synthesis/{rtl_id}/syn/{effort}"
    graph_output_file = data_dir / f"netlist/graph/{rtl_id}_{effort}.pkl"
    bboxes = [cg.io.BlackBox("f", ["CK", "D"], ["Q"])] + cg.io.genus_flops + cg.io.dc_flops

    try:
        remove_module_suffix(syn_path / "syn.v", syn_path / "syn_copy.v")
        circuit = cg.from_file(syn_path / "syn_copy.v", blackboxes=bboxes)
        anonymized_graph = anonymize_graph(circuit.graph)
        with open(graph_output_file, 'wb') as f:
            pkl.dump(anonymized_graph, f)
        os.remove(syn_path / "syn_copy.v")
    except Exception:
        if os.path.exists(syn_path / "syn_copy.v"):
            os.remove(syn_path / "syn_copy.v")
        return (rtl_id, effort)
    
    return None


def generate_netlist_graph_multithreads(num_cores: int, data_dir: Path) -> None:
    """
    Collect netlist graphs by processing Verilog files in parallel and saving results.

    Args:
        num_cores (int): Number of CPU cores to use for multiprocessing.
        data_dir (Path): Base directory containing synthesis results and output folders.
    """
    with open(data_dir / "synthesis/synthesis_result.pkl", 'rb') as f:
        success, _, _ = pkl.load(f)

    netlist_dir = data_dir / "netlist"
    netlist_graph_dir = netlist_dir / "graph"
    netlist_graph_dir.mkdir(parents=True, exist_ok=True)

    fail = []
    efforts = [f'{e1}_{e2}_{e3}' for e1, e2, e3 in product(("low", "medium", "high"), repeat=3)]
    inputs = [(rtl_id, effort, data_dir) for rtl_id in success for effort in efforts]

    with mp.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_netlist, inputs), total=len(inputs), desc="Processing Netlists"))
    fail = [res for res in results if res is not None]

    with open(netlist_dir / "netlist_graphgen_fail.pkl", 'wb') as f:
        pkl.dump(fail, f)

    print(f"Graph processing complete. Failures: {len(fail)}")


def main(
    num_cores: int = typer.Option(4, help="Number of CPU cores to use for multiprocessing."),
    data_dir: Path = typer.Option(..., help="Path to the base directory containing synthesis results.")
):
    """
    Entry point to generate and process netlist graphs from synthesis results.

    Args:
        num_cores (int): Number of CPU cores to utilize for parallel processing.
        data_dir (Path): Base directory containing synthesis result and output folders.
    """
    generate_netlist_graph_multithreads(num_cores, data_dir)
    print("Netlist graph generattion complete!")


if __name__ == "__main__":
    typer.run(main)
