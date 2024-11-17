from typing import List, Optional
import fire
from llama import Dialog, Llama
import json
import pickle as pkl
from tqdm import tqdm
import warnings
import os
os.environ['XDG_CACHE_HOME'] = '/home/weili3/.cache'
os.makedirs('/home/weili3/.cache', exist_ok=True)
warnings.filterwarnings("ignore", category=UserWarning)

categories = """
1. Encryption Units: Modules that handle encryption or cryptographic functions.
2. Data Path Units: Modules involved in data movement, selection, or manipulation (e.g., multiplexers, shifters).
3. Control Logic Units: Modules responsible for control flow or decision-making in systems (e.g., state machines).
4. Arithmetic Units: Modules performing arithmetic operations (e.g., adders, subtractors).
5. Communication Protocol Units: Modules implementing communication protocols (e.g., UART, SPI).
6. Signal Processing Units: Modules used for signal transformation or filtering.
7. Clock Management Units: Modules managing clock signals and synchronization.
8. Other Units: Modules not fitting the above categories.
"""

example_one = r"""
Description: "This module is a 4-bit adder with carry-in and carry-out. The module has two 4-bit inputs, a single carry-in input, and a single carry-out output. The output is the sum of the two inputs plus the carry-in."
Verilog: "module adder (\n    input [3:0] a,\n    input [3:0] b,\n    input cin,\n    output cout,\n    output [3:0] sum\n);\n\n    assign {cout, sum} = a + b + cin;\n\nendmodule"
Response: "Arithmetic Units"
"""

example_two = r"""
Description: "This module is a 2-to-1 multiplexer designed using Verilog. The module has two input ports and one output port. The output is the value of the first input port if the select input is 0, and the value of the second input port if the select input is 1. The design is implemented using only NAND gates."
Verilog: "module mux_2to1 (\n    input a,\n    input b,\n    input select,\n    output reg out\n);\n\n  wire nand1, nand2, nand3, nand4;\n\n  assign nand1 = ~(a & select);\n  assign nand2 = ~(b & ~select);\n  assign nand3 = ~(nand1 & nand2);\n  assign nand4 = ~(nand3 & nand3);\n\n  always @ (nand4) begin\n    out <= ~nand4;\n  end\n\nendmodule"
Response: "Data Path Units"
"""


system_setting = f"""\
You are a professional VLSI digital design engineer. Categorize the following RTL (Register Transfer Level) design descriptions and Verilog code pairs into one of the functional categories below. The response should only contain the most relevant function category:

Functional Categories:{categories}
Please reply with only the functional category name.

Examples:
1.{example_one}2.{example_two}
Now categorize the following RTL description and Verilog code pair:
"""


def predict_function_category(
    description,
    verilog,
    generator,
    temperature,
    top_p,
    max_gen_len: Optional[int] = None,
):
    dialogs: List[Dialog] = [
        [
            {"role": "system", "content": system_setting},
            {"role": "user", "content": f"""Description: "{description}"\nVerilog: "{verilog}" """}
        ]
    ]
    try:
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=True,
        )
        reply = results[0]['generation']['content']
        tokens =  results[0]['tokens']
        logprob =  results[0]['logprobs']
        return reply, tokens, logprob
    except:
        return None, None

def most_frequent(L):
    return max(set(L), key=L.count)

def main(
    json_path: str, # path to the RTL json file
    data_file: str, # path to the original RTL json dataset
    ckpt_dir: str, # LLAMA3 checkpoint directory
    tokenizer_path: str, # LLAMA3 tokenizer path
    temperature: float = 0,
    top_p: float = 0,
    max_seq_len: int = 8192,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    with open(json_path) as f:
       json_content = json.load(f)
    # verilog_list = []
    # with open(data_file) as f:
    #     for _, line in enumerate(f):
    #         if not line.strip():
    #             continue
    #         data = json.loads(line)
    #         verilog_list.append(data.get("Response", [""])[0])
    with open(data_file, "rb") as f:
        verilog_list = pkl.load(f)
    description_verilog_list = [(int(key), json_content[key]['description'], verilog_list[int(key)]) for key in json_content.keys() if json_content[key]['synthesis_status']]

    i = 0
    res = []
    for i, (idx, description, verilog) in tqdm(enumerate(description_verilog_list)):
        reply, tokens, logprob = predict_function_category(description, verilog, generator, temperature, top_p, max_gen_len)
        res.append((i, idx, reply, tokens, logprob))
        # occur = set()
        # L = []
        # for _ in range(10):
        #     reply = predict_function_category(description, verilog, generator, temperature, top_p, max_gen_len)
        #     occur.add(reply)
        #     L.append(reply)
        # if len(occur) != 1:
        #     res.append((i, idx, False, most_frequent(L)))
        # else:
        #     res.append((i, idx, True, L[0]))
        with open("/home/weili3/VLSI-LLM/data_collection/llama3-70b_label.pkl", "wb") as f:
            pkl.dump(res, f)

if __name__ == "__main__":
    fire.Fire(main)

"""
torchrun --nproc_per_node 2 /home/weili3/VLSI-LLM/data_collection/identify_function_label.py \
    --json_path /home/weili3/VLSI-LLM/data_collection/rtl_LLM11442.json \
    --data_file /home/weili3/VLSI-LLM/data_collection/verilog.pkl \
    --ckpt_dir /home/weili3/llama3/Meta-Llama-3-70B-Instruct-2-shards/ \
    --tokenizer_path /home/weili3/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model \
    --max_seq_len 8192 \
    --max_batch_size 1
"""

#100 - 7  [(60, 92), (65, 100), (66, 101), (73, 112), (78, 119), (81, 125), (90, 136)]
#100 - 11 [(19, 29), (60, 92), (63, 98), (65, 100), (66, 101), (71, 109), (73, 112), (74, 114), (78, 119), (81, 125), (90, 136)], [(101, 151), (103, 154), (115, 176), (119, 180), (120, 185), (125, 192), (137, 205), (149, 219), (153, 228), (155, 230), (156, 231), (164, 245), (168, 249), (181, 272), (188, 281), (192, 289), (194, 291), (197, 294)]