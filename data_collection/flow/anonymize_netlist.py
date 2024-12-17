import re
import os
import typer
from tqdm import tqdm
from pathlib import Path

def anonymize_netlist(original_verilog_folder, anonymized_verilog_folder):
    """
    Anonymize Verilog netlist files by renaming module names and their instantiations.

    Args:
        original_verilog_folder (str or Path): Path to the folder containing the original Verilog files.
        anonymized_verilog_folder (str or Path): Path to the folder where anonymized Verilog files will be saved.

    Functionality:
        - Reads all Verilog files (`.v` extension) from the original folder.
        - Replaces module names with anonymized names (e.g., anonymized_module_0, anonymized_module_1, etc.).
        - Ensures module name changes are consistent across the file (including module declarations and instantiations).
        - Saves the anonymized versions of the files in a new folder while preserving the original files.
    """
    
    original_verilog_folder = Path(original_verilog_folder)
    anonymized_verilog_folder = Path(anonymized_verilog_folder)
    anonymized_verilog_folder.mkdir(parents=True, exist_ok=True)

    def anonymize_modules(verilog_code):
        """
        Replace module names and their instantiations in Verilog code with anonymized names.

        Args:
            verilog_code (str): The Verilog source code as a string.

        Returns:
            tuple: (anonymized_code (str), module_mapping (dict))
                - anonymized_code: Verilog code with module names replaced.
                - module_mapping: A dictionary mapping original module names to anonymized names.
        """
        module_mapping = {}
        counter = 0
        pattern = r'(\bmodule\s+)(\w+)(\s*\()'

        def replace_module_name(match):
            """
            Replace the original module name with an anonymized name during regex substitution.
            
            Args:
                match: Regex match object containing the module name.
                
            Returns:
                str: Updated module declaration with the anonymized name.
            """
            nonlocal counter
            original_module_name = match.group(2)
            anonymized_module_name = f"anonymized_module_{counter}"
            module_mapping[original_module_name] = anonymized_module_name
            counter += 1
            return f"{match.group(1)}{anonymized_module_name}{match.group(3)}"
        
        anonymized_code = re.sub(pattern, replace_module_name, verilog_code, flags=re.MULTILINE)

        for original_name, anonymized_name in module_mapping.items():
            instantiation_pattern = rf'\b{original_name}\b'  # Match exact module name for replacement.
            anonymized_code = re.sub(instantiation_pattern, anonymized_name, anonymized_code)

        return anonymized_code, module_mapping

    verilog_files = [f for f in os.listdir(original_verilog_folder) if f.endswith('.v')]
    
    for filename in tqdm(verilog_files, desc="Anonymizing Verilog Files"):
        with open(original_verilog_folder / filename, 'r') as f:
            content = f.read()
        code, _ = anonymize_modules(content)
        with open(anonymized_verilog_folder / filename, 'w') as f:
            f.write(code)

def main(original_verilog_folder: str, anonymized_verilog_folder: str):
    anonymize_netlist(original_verilog_folder, anonymized_verilog_folder)

if __name__ == "__main__":
    typer.run(main)