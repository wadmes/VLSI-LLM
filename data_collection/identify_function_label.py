from typing import List, Optional
import fire
from llama import Dialog, Llama
import json
import pickle as pkl
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(
    json_path: str, # path to the RTL json file
    ckpt_dir: str, # LLAMA3 checkpoint directory
    tokenizer_path: str, # LLAMA3 tokenizer path
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    
    with open(json_path) as f:
       json_content = json.load(f)
    
    res = []
    error = []

    for i in tqdm(json_content.keys()):
        dialogs: List[Dialog] = [
            [
                {
                   "role": "user", "content":
                   """
                    I will provide you with the instruction of an RTL design. Based on its functionality, structure, and components used, output the most likely category index from the following list as a single digit:
                    \n1. Encryption Units
                    \n2. Data Path Units
                    \n3. Control Logic Unit
                    \n4. Arithmetic Unit
                    \n5. Communication Protocol Unit
                    \n6. Signal Processing Unit
                    \n7. Clock Management Unit
                    \n8. Others
                    \nProvide only a single digit (1-8) representing the category index.
                    \n
                    \nHere are two examples:
                    \n1:
                    \nInstruction: Design a module that can detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module should have two input ports: a clock input and an 8-bit binary input port. The output port should be an 8-bit binary vector that represents the input value one cycle after the edge is detected. The module must be designed using a counter and a comparator.
                    \nExample Response: 3
                    \n2:
                    \nInstruction: You are tasked with designing a Verilog module that generates a pseudo-random bit sequence (PRBS) using a linear-feedback shift register (LFSR). The module should have the following inputs and outputs:\n\nInputs:\n- `clk_i`: clock input\n- `clk_en_i`: clock enable input\n- `rst_i`: synchronous reset input\n- `prbs_seed_i`: initial seed for the LFSR\n- `phy_if_empty`: IN_FIFO empty flag\n- `prbs_rdlvl_start`: PRBS read leveling start\n\nOutputs:\n- `prbs_o`: generated PRBS sequence\n\nThe module should use a 64-bit LFSR shift register to generate the PRBS sequence. The LFSR should be reset to the initial seed value provided by `prbs_seed_i` on reset or when `phy_if_empty` is high. The module should generate a PRBS reset signal to ensure that the PRBS sequence repeats after every 2^64 samples. The PRBS reset signal should be asserted on the last sample of the sequence.\n\nThe module should support read leveling by asserting `prbs_rdlvl_start` when the read leveling process starts. During read leveling, the PRBS sequence should continue to be generated, but should not be output on `prbs_o`. Once read leveling is complete, the module should resume outputting the PRBS sequence on `prbs_o`
                    \nExample Response: 5
                    \n
                    \nNow categorize this one:
                    \ninstruction:
                    \n
                    """
                    + json_content[i]['instruction']
                },
            ]
        ]
        try:
            results = generator.chat_completion(
                dialogs,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            res.append((i, results[0]['generation']['content']))
        except:
            error.append(i)


    print("Total error: ", len(error))
    with open("/home/weili3/VLSI-LLM/data_collection/label.pkl", "wb") as f:
        pkl.dump((res, error), f)

if __name__ == "__main__":
    fire.Fire(main)

"""
torchrun --nproc_per_node 2 /home/weili3/VLSI-LLM/data_collection/identify_function_label.py
    --ckpt_dir /home/weili3/llama3/Meta-Llama-3-70B-Instruct-2-shards/
    --tokenizer_path /home/weili3/llama3/Meta-Llama-3-70B-Instruct/tokenizer.model
    --max_seq_len 8192 --max_batch_size 1
    --json_path /home/weili3/VLSI-LLM/data_collection/rtl_data/rtl.json
"""