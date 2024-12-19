"""
data generation step 9
This script organizes and documents successful netlist files from synthesis results.
It filters out failed netlist processes, copies the associated Verilog and log files
into designated directories, and generates a JSON record of successful netlist metadata.
"""

import json
import shutil
import pickle as pkl
from tqdm import tqdm
from pathlib import Path
from itertools import product
import typer

def main(
    data_dir: Path = typer.Option(..., help="Path to the base data directory.")
) -> None:
    """
    Generate a JSON record of successful netlist files and copy Verilog and log files 
    to designated directories.

    Args:
        data_dir (Path): Path to the base directory containing synthesis results 
                        and netlist-related folders.
    """
    with open(data_dir / "synthesis/synthesis_result.pkl", 'rb') as f:
        success, _, _ = pkl.load(f)
    with open(data_dir / "netlist_data/netlist_graphgen_fail.pkl", 'rb') as f:
        fail = pkl.load(f)

    netlist_verilog_dir = data_dir / "netlist_data/verilog"
    synthesis_log_dir = data_dir / "netlist_data/synthesis_log"
    netlist_verilog_dir.mkdir(parents=True, exist_ok=True)
    synthesis_log_dir.mkdir(parents=True, exist_ok=True)

    records_dict = {}
    idx = 0
    efforts = [f"{e1}_{e2}_{e3}" for e1, e2, e3 in product(("low", "medium", "high"), repeat=3)]
    for rtl_id, effort in tqdm([(rtl_id, effort) for rtl_id in success for effort in efforts], desc="Generating Netlist JSON"):
        if (rtl_id, effort) not in fail:
            syn_path = data_dir / f"synthesis/{rtl_id}/syn/{effort}"
            verilog_file = syn_path / "syn.v"
            log_file = syn_path / "genus.log"
            dest_verilog = netlist_verilog_dir / f"{rtl_id}_{effort}.v"
            dest_log = synthesis_log_dir / f"{rtl_id}_{effort}.log"
            if verilog_file.exists() and log_file.exists():
                shutil.copy(verilog_file, dest_verilog)
                shutil.copy(log_file, dest_log)
            records_dict[idx] = {
                'rtl_id': rtl_id,
                'synthesis_efforts': effort,
            }
            idx += 1

    with open(data_dir / "netlist_data/netlist.json", mode='w') as file:
        json.dump(records_dict, file, indent=4)

    print(f"Netlist JSON generated successfully!")

if __name__ == "__main__":
    typer.run(main)
