"""
The script is a example that reads RTL json file, and translate the `rtl_instruction` into `rtl_description` using LLAMA3.
"""

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
    """
    [
    load json file, the json file is a list of dictionaries, example is:
{
    "rtl_id": 13002,
    "rtl": "module note2dds(clk, note, adder);\n\ninput wire clk;\ninput wire [8:0] note;\noutput [31:0] adder;\n\nreg [31:0] adder_tbl [15:0];\nreg [3:0] addr;\nreg [3:0] divider;\n\ninitial begin\n   addr <= 4'd0;\n   divider <= 4'd0;\n   adder_tbl[ 4'd0] <= 32'd01438301;\n   adder_tbl[ 4'd1] <= 32'd01523827;\n   adder_tbl[ 4'd2] <= 32'd01614439;\n   adder_tbl[ 4'd3] <= 32'd01710438;\n   adder_tbl[ 4'd4] <= 32'd01812146;\n   adder_tbl[ 4'd5] <= 32'd01919902;\n   adder_tbl[ 4'd6] <= 32'd02034065;\n   adder_tbl[ 4'd7] <= 32'd02155017;\n   adder_tbl[ 4'd8] <= 32'd02283161;\n   adder_tbl[ 4'd9] <= 32'd02418925;\n   adder_tbl[4'd10] <= 32'd02562762;\n   adder_tbl[4'd11] <= 32'd02715152;\n   adder_tbl[4'd12] <= 32'd0;\n   adder_tbl[4'd13] <= 32'd0;\n   adder_tbl[4'd14] <= 32'd0;\n   adder_tbl[4'd15] <= 32'd0;\nend\n\nassign adder = adder_tbl[addr] >> divider;\n\nwire [5:0] diap   = (note <  12) ? 6'd00 :\n                     (note <  24) ? 6'd01 :\n                     (note <  36) ? 6'd02 :\n                     (note <  48) ? 6'd03 :\n                     (note <  60) ? 6'd04 :\n                     (note <  72) ? 6'd05 :\n                     (note <  84) ? 6'd06 :\n                     (note <  96) ? 6'd07 :\n                     (note < 108) ? 6'd08 :\n                     (note < 120) ? 6'd09 : \n                     (note < 132) ? 6'd10 : \n                     (note < 144) ? 6'd11 : \n                     (note < 156) ? 6'd12 : \n                     (note < 168) ? 6'd13 : \n                     (note < 180) ? 6'd14 : \n                     (note < 192) ? 6'd15 : \n                     (note < 204) ? 6'd16 : \n                     (note < 216) ? 6'd17 : \n                     (note < 228) ? 6'd18 : \n                     (note < 240) ? 6'd19 : \n                     (note < 252) ? 6'd20 : \n                     (note < 264) ? 6'd21 : \n                     (note < 276) ? 6'd22 : \n                     (note < 288) ? 6'd23 : \n                     (note < 300) ? 6'd24 : \n                     (note < 312) ? 6'd25 : \n                     (note < 324) ? 6'd26 : \n                     (note < 336) ? 6'd27 : \n                     (note < 348) ? 6'd28 : \n                     (note < 360) ? 6'd29 : \n                     (note < 372) ? 6'd30 : \n                     (note < 384) ? 6'd31 : \n                     (note < 396) ? 6'd32 : \n                     (note < 408) ? 6'd33 : \n                     (note < 420) ? 6'd34 : \n                     (note < 432) ? 6'd35 : \n                     (note < 444) ? 6'd36 : \n                     (note < 456) ? 6'd37 : \n                     (note < 468) ? 6'd38 : \n                     (note < 480) ? 6'd39 : \n                     (note < 492) ? 6'd40 : \n                     (note < 504) ? 6'd41 : 6'd042 ;\n\nwire [6:0] c_addr = note - (diap * 4'd012); \n                         \nalways @ (posedge clk) begin\n    case (c_addr[3:0])\n        4'd0: addr <= 4'd0;\n        4'd1: addr <= 4'd1;\n        4'd2: addr <= 4'd2;\n        4'd3: addr <= 4'd3;\n        4'd4: addr <= 4'd4;\n        4'd5: addr <= 4'd5;\n        4'd6: addr <= 4'd6;\n        4'd7: addr <= 4'd7;\n        4'd8: addr <= 4'd8;\n        4'd9: addr <= 4'd9;\n        4'd10: addr <= 4'd10;\n        4'd11: addr <= 4'd11;\n        default: addr <= 4'd0;\n    endcase\n    \n    case (diap)\n        6'd0: divider <= 4'd42;\n        6'd1: divider <= 4'd38;\n        6'd2: divider <= 4'd34;\n        6'd3: divider <= 4'd30;\n        6'd4: divider <= 4'd26;\n        6'd5: divider <= 4'd22;\n        6'd6: divider <= 4'd18;\n        6'd7: divider <= 4'd14;\n        6'd8: divider <= 4'd10;\n        6'd9: divider <= 4'd6;\n        6'd10: divider <= 4'd2;\n        6'd11: divider <= 4'd0;\n        6'd12: divider <= 4'd0;\n        6'd13: divider <= 4'd0;\n        6'd14: divider <= 4'd0;\n        6'd15: divider <= 4'd0;\n        6'd16: divider <= 4'd0;\n        6'd17: divider <= 4'd0;\n        6'd18: divider <= 4'd0;\n        6'd19: divider <= 4'd0;\n        6'd20: divider <= 4'd0;\n        6'd21: divider <= 4'd0;\n        6'd22: divider <= 4'd0;\n        6'd23: divider <= 4'd0;\n        6'd24: divider <= 4'd0;\n        6'd25: divider <= 4'd0;\n        6'd26: divider <= 4'd0;\n        6'd27: divider <= 4'd0;\n        6'd28: divider <= 4'd0;\n        6'd29: divider <= 4'd0;\n        6'd30: divider <= 4'd0;\n        6'd31: divider <= 4'd0;\n        6'd32: divider <= 4'd0;\n        6'd33: divider <= 4'd0;\n        6'd34: divider <= 4'd0;\n        6'd35: divider <= 4'd0;\n        6'd36: divider <= 4'd0;\n        6'd37: divider <= 4'd0;\n        6'd38: divider <= 4'd0;\n        6'd39: divider <= 4'd0;\n        6'd40: divider <= 4'd0;\n        6'd41: divider <= 4'd0;\n        default: divider <= 4'd0;\n    endcase\nend\n\nendmodule",
    "rtl_instruction": "\n\nYou are tasked with designing a Verilog module that converts musical notes into a frequency that can be used to generate sound. The module should take in a clock signal, a 9-bit input representing a musical note, and output a 32-bit value representing the frequency of the note.\n\nThe input note is represented as a value between 0 and 511, where 0 represents a C0 note and 511 represents a B9 note. The module should be able to handle notes within this range.\n\nThe output frequency should be calculated using a lookup table and a divider. The lookup table is an array of 16 32-bit values, each representing the frequency of a note. The divider is a value between 0 and 42 that is used to adjust the frequency of the note.\n\nThe module should also be able to divide the input note by 12, which corresponds to a one octave decrease in frequency. Additionally, the module should be able to divide the input note by 48 and 192, which correspond to a decrease in frequency by one and two octaves, respectively.\n\nYour task is to implement a Verilog module that takes in a clock signal and a 9-bit input representing a musical note, and outputs a 32-bit value representing the frequency of the note.\n\n",
    "rtl_description": "",
    "rtl_data_flow_filename": "13002.pkl"
},
{
    "rtl_id": 13003,
    "rtl": "module prbs_gen (\n    input clk_i,\n    input clk_en_i,\n    input rst_i,\n    input [63:0] prbs_seed_i,\n    input phy_if_empty,\n    input prbs_rdlvl_start,\n    output reg [63:0] prbs_o\n);\n\n    // Define LFSR polynomial\n    parameter LFSR_POLY = 64'hAD6F7036CCBFE794;\n\n    // Define PRBS sequence length in cycles\n    parameter PRBS_SEQ_LEN_CYCLES = 128;\n\n    // Define PRBS sequence length in bits\n    parameter PRBS_SEQ_LEN_BITS = $clog2(PRBS_SEQ_LEN_CYCLES);\n\n    // Define PRBS reset signal length in cycles\n    parameter PRBS_RESET_CYCLES = 2;\n\n    // Define PRBS reset signal length in bits\n    parameter PRBS_RESET_BITS = $clog2(PRBS_RESET_CYCLES);\n\n    // Define PRBS reset signal counter\n    reg [PRBS_RESET_BITS-1:0] prbs_reset_cnt;\n\n    // Define PRBS sequence length counter\n    reg [PRBS_SEQ_LEN_BITS-1:0] prbs_seq_cnt;\n\n    // Define LFSR shift register\n    reg [63:0] lfsr;\n\n    // Define PRBS sequence output enable\n    reg prbs_output_enable;\n\n    // Reset LFSR and counters on reset or when FIFO is empty\n    always @(posedge clk_i) begin\n        if (rst_i || phy_if_empty) begin\n            lfsr <= prbs_seed_i;\n            prbs_reset_cnt <= 0;\n            prbs_seq_cnt <= 0;\n            prbs_output_enable <= 1'b0;\n        end\n    end\n\n    // Generate PRBS reset signal to ensure sequence repeats after 2^64 samples\n    always @(posedge clk_i) begin\n        if (rst_i || ~clk_en_i) begin\n            prbs_reset_cnt <= 0;\n        end else if (clk_en_i && (~phy_if_empty || ~prbs_rdlvl_start)) begin\n            if (prbs_reset_cnt == PRBS_RESET_CYCLES-1) begin\n                lfsr <= prbs_seed_i;\n                prbs_seq_cnt <= 0;\n                prbs_output_enable <= 1'b0;\n            end else if (prbs_seq_cnt == PRBS_SEQ_LEN_CYCLES-1) begin\n                prbs_reset_cnt <= prbs_reset_cnt + 1;\n            end\n        end\n    end\n\n    // Generate PRBS sequence\n    always @(posedge clk_i) begin\n        if (rst_i || ~clk_en_i) begin\n            prbs_o <= 0;\n        end else if (clk_en_i && (~phy_if_empty || ~prbs_rdlvl_start)) begin\n            if (prbs_seq_cnt == PRBS_SEQ_LEN_CYCLES-1) begin\n                prbs_seq_cnt <= 0;\n                prbs_output_enable <= 1'b1;\n            end else begin\n                prbs_seq_cnt <= prbs_seq_cnt + 1;\n                prbs_output_enable <= 1'b0;\n            end\n\n            if (prbs_output_enable) begin\n                prbs_o <= lfsr[63];\n            end\n\n            lfsr <= {lfsr[62:0], lfsr[0] ^ lfsr[3] ^ lfsr[7] ^ lfsr[10] ^ lfsr[11] ^ lfsr[13] ^ lfsr[14] ^ lfsr[15] ^ lfsr[16] ^ lfsr[17] ^ lfsr[21] ^ lfsr[22] ^ lfsr[23] ^ lfsr[26] ^ lfsr[28] ^ lfsr[31] ^ lfsr[32] ^ lfsr[33] ^ lfsr[34] ^ lfsr[35] ^ lfsr[36] ^ lfsr[40] ^ lfsr[41] ^ lfsr[42] ^ lfsr[45] ^ lfsr[46] ^ lfsr[52] ^ lfsr[53] ^ lfsr[54] ^ lfsr[56] ^ lfsr[58] ^ lfsr[59] ^ lfsr[63]};\n        end\n    end\n\nendmodule",
    "rtl_instruction": "\n\nYou are tasked with designing a Verilog module that generates a pseudo-random bit sequence (PRBS) using a linear-feedback shift register (LFSR). The module should have the following inputs and outputs:\n\nInputs:\n- `clk_i`: clock input\n- `clk_en_i`: clock enable input\n- `rst_i`: synchronous reset input\n- `prbs_seed_i`: initial seed for the LFSR\n- `phy_if_empty`: IN_FIFO empty flag\n- `prbs_rdlvl_start`: PRBS read leveling start\n\nOutputs:\n- `prbs_o`: generated PRBS sequence\n\nThe module should use a 64-bit LFSR shift register to generate the PRBS sequence. The LFSR should be reset to the initial seed value provided by `prbs_seed_i` on reset or when `phy_if_empty` is high. The module should generate a PRBS reset signal to ensure that the PRBS sequence repeats after every 2^64 samples. The PRBS reset signal should be asserted on the last sample of the sequence.\n\nThe module should support read leveling by asserting `prbs_rdlvl_start` when the read leveling process starts. During read leveling, the PRBS sequence should continue to be generated, but should not be output on `prbs_o`. Once read leveling is complete, the module should resume outputting the PRBS sequence on `prbs_o`.\n\n",
    "rtl_description": "",
    "rtl_data_flow_filename": "13003.pkl"
},
    """
    # Now, load json file as a list of dictionaries
    with open(json_path) as f:
       json_content = json.load(f)
    
    error = []
    res = []
    for i in tqdm(json_content.keys()):
        if json_content[i]['description'] == "":
            dialogs: List[Dialog] = [
                [
                    {"role": "user", "content": """Given a design instruction, change it into a tone of description. Do not change or add any details.  \n
                    Here are two examples. \n
                    Instruction: Design a module that can detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module should have two input ports: a clock input and an 8-bit binary input port. The output port should be an 8-bit binary vector that represents the input value one cycle after the edge is detected. The module must be designed using a counter and a comparator.
                    \n Example description: This module is designed to detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module has two input ports: a clock input (`clk`) and an 8-bit binary input port (`in`). The output port (`out`) is an 8-bit binary vector that represents the input value one cycle after the edge is detected. The design uses a counter and a comparator to achieve this functionality.
                    \n Instruction: Please act as a professional Verilog designer. Design a pipelined module that implements a 4-to-2 priority encoder. The module should have four 1-bit inputs (I0, I1, I2, I3) and two 2-bit outputs (O0, O1). The output should be the binary encoding of the highest-priority input that is asserted. If multiple inputs are asserted, the output should correspond to the input with the highest index number (i.e., the last asserted input in the list). Use pipeline structure to achieve this functionality.
                    \n Example description: This design is a pipelined 4-to-2 priority encoder module. The module has four 1-bit inputs (`I0`, `I1`, `I2`, `I3`) and two 2-bit outputs (`O0`, `O1`). The output is the binary encoding of the highest-priority input that is asserted. If multiple inputs are asserted, the output corresponds to the input with the highest index number. The design uses a pipeline structure to implement this functionality.
                    \n Now, please change this instruction directly (do not include any pre-fix like `here is a rewritten description): """ + json_content[i]['instruction']},
                ]
            ]
            try:
                results = generator.chat_completion(
                    dialogs,
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                print(results[0]['generation']['content'])
                # save the results as `rtl_description` in the json file
                json_content[i]['description'] = results[0]['generation']['content']
            except:
                res += ["" for _ in range(100)]
                error.append(i)


    print("Total error: ", len(error))
    # save the results as a json file
    new_json_path = json_path.replace(".json", "_with_desc.json")
    with open(new_json_path, "w") as f:
        json.dump(json_content, f)



if __name__ == "__main__":
    fire.Fire(main)
