# combine the MGverilog.csv and RTLCoder.csv, while make the netlist_id in MGverilog.csv + 1000000

import pandas as pd
import os

MGverilog = pd.read_csv('MGVerilog.csv')
RTLCoder = pd.read_csv('RTLCoder.csv')
MGverilog['id'] = MGverilog['id'] + 1000000
result = pd.concat([MGverilog, RTLCoder], ignore_index=True)
result.to_csv('BRIDGES.csv', index=False)