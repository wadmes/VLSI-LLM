# Data generation and process for VLSI-LLM
This directory contains the code and guide documents that automates the data collection process.

## Data Generation
TODO (Add the data generation process, (scripts, how to run, etc))


## Data Format
The processed data includes three files:

1. `RTL.csv`: contains the RTL data sheet, columns are:
    + rtl_id (int): the unique id of the RTL
    + rtl (str): the RTL content
    + rtl_instruction (str): the instruction to generate RTL for LLM
    + rtl_description (str): the function description of the RTL
    + rtl_data_flow_filename (str): the filename of the data flow graph (networkx format)

2. `netlist.csv`: contains the netlist data sheet, columns are:
    + netlist_id (int): the unique id of the netlist
    + netlist (str): the netlist content (.v file content)
    + rtl_id (int): the id of the corresponding RTL
    + synthesis_command (str): the synthesis command to generate the netlist
    + synthesis_log (str): the synthesis log when generating the netlist
    + netlist_graph_filename (str): the filename of the netlist graph (networkx format)

3. `data_flow_graph/`: contains the data flow graph files, each file is a networkx graph object in pickle format.

4. `netlist_graph/`: contains the netlist graph files, each file is a networkx graph object in pickle format.
