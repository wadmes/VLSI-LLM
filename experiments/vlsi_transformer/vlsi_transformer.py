import json
import logging
import argparse
from typing import List, Dict
from vllm import LLM, SamplingParams

# Set up argument parser
parser = argparse.ArgumentParser(description="Transform RTL fields using vLLM.")
parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use for inference")
parser.add_argument("--json_file", default="rtl.json", help="Input JSON file")
parser.add_argument("--max_batch_size", type=int, default=1000, help="Maximum batch size for inference")
parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
parser.add_argument("--max_tokens", type=int, default=200, help="Maximum number of tokens to generate")
parser.add_argument("--log_file", default="vlsi_processing.log", help="Log file path")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for inference")
parser.add_argument("--input_field", default="instruction", help="Input field in JSON to convert from [instruction, description, verilog]")
parser.add_argument("--output_field", default="description", help="Output field in JSON to convert to [instruction, description, verilog]")
args = parser.parse_args()

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=args.log_file,
                    filemode='w')
logger = logging.getLogger(__name__)

# Add a stream handler for error messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
logger.addHandler(console_handler)

def write_responses(new_data: Dict, output_file: str):
    logger.info(f"Writing response to {output_file}")
    try:
        with open(output_file, 'r+') as file:
            file.seek(0)
            content = file.read().strip()
            if not content:
                # File is empty, write new data directly
                json.dump(new_data, file, indent=2)
            else:
                # File contains data, load it and update
                file.seek(0)
                existing_data = json.load(file)
                existing_data.update(new_data)
                file.seek(0)
                file.truncate()
                json.dump(existing_data, file, indent=2)
        logger.info(f"Response written successfully to {output_file}")
    except Exception as e:
        logger.error(f"Error writing to {output_file}: {str(e)}")

def send_request(prompts: List[List[Dict]], llm: LLM, sampling_params: SamplingParams) -> List:
    logger.info(f"Sending batch request with {len(prompts)} prompts")
    try:
        outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        logger.info(f"Received {len(outputs)} responses")
        return outputs
    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}")
        return []

def join_requests(json_content: Dict, completed_requests: Dict, json_file: str, input_field: str, output_field: str):
    updated_items = {}
    for rtl_id, data in completed_requests.items():
        json_content[str(rtl_id)][output_field] = data['description']
        updated_items[str(rtl_id)] = json_content[str(rtl_id)]
    write_responses(updated_items, json_file)

def create_prompt(input_text: str, system_message: str, user_message: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"{user_message} {input_text}"
        },
    ]

def process_batch(prompts: List[List[Dict]], request_ids: List[str], llm: LLM, sampling_params: SamplingParams) -> Dict:
    logger.info(f"Processing batch of {len(prompts)} prompts")
    outputs = send_request(prompts, llm, sampling_params)
    completed_requests = {}
    for output, request_id in zip(outputs, request_ids):
        try:
            response_text = output.outputs[0].text
            completed_requests[request_id] = {"description": response_text}
        except Exception as e:
            logger.error(f"Failed to process output for request ID {request_id}: {str(e)}")
    logger.info(f"Completed processing batch of {len(prompts)} prompts")
    return completed_requests

def generate_system_message(input_type, output_type):
    # Validate input and output types
    valid_types = ["instruction", "description", "verilog"]
    if input_type not in valid_types or output_type not in valid_types:
        raise ValueError(f"input_type and output_type must be one of {valid_types}")

    # Define task based on input and output types
    task_mapping = {
        ("instruction", "description"): "Rewrite a given design instruction as a 1-4 sentence descriptive statement.",
        ("description", "instruction"): "Convert a given design description into a clear and concise design instruction.",
        ("instruction", "verilog"): "Translate a given design instruction into functional verilog code.",
        ("verilog", "instruction"): "Generate a design instruction based on the provided verilog code.",
        ("description", "verilog"): "Develop verilog code based on the provided design description.",
        ("verilog", "description"): "Create a descriptive statement that explains the functionality of the provided verilog code."
    }

    task = task_mapping.get((input_type, output_type))
    if not task:
        raise ValueError(f"Transformation from {input_type} to {output_type} is not supported.")

    # Define examples based on transformation
    instruction_example = '''
Design a module that can detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module should have two input ports: a clock input and an 8-bit binary input port. The output port should be an 8-bit binary vector that represents the input value one cycle after the edge is detected. The module must be designed using a counter and a comparator.
'''

    description_example = '''
This module is designed to detect any edge in an 8-bit binary vector and output the binary value of the vector one cycle after the edge is detected. The module has two input ports: a clock input (`clk`) and an 8-bit binary input port (`in`). The output port (`out`) is an 8-bit binary vector that represents the input value one cycle after the edge is detected. The design uses a counter and a comparator to achieve this functionality.
'''

    verilog_code_example = '''
module EdgeDetector (
    input wire clk,
    input wire [7:0] in,
    output reg [7:0] out
);

    // Register to hold the previous input value
    reg [7:0] prev_in;
    
    // Counter to delay the output by one cycle after edge detection
    reg [3:0] counter;
    wire edge_detected;

    // Comparator to detect any edge in the 8-bit input vector
    assign edge_detected = (in != prev_in);

    always @(posedge clk) begin
        // Update previous input
        prev_in <= in;
        
        if (edge_detected) begin
            counter <= 4'd1; // Start the counter
        end else if (counter > 0) begin
            counter <= counter + 1;
        end

        // Output the input value one cycle after edge detection
        if (counter == 4'd2) begin
            out <= prev_in;
            counter <= 4'd0; // Reset the counter
        end
    end

endmodule
'''
    examples = {
        ("instruction", "description"): f'''
            EXAMPLES:

            Original instruction:
            {instruction_example}

            Example description:
            {description_example}
            ''',
        ("description", "instruction"): f'''
            EXAMPLES:

            Original description:
            {description_example}

            Example instruction:
            {instruction_example}
            ''',
        ("instruction", "verilog"): f'''
            EXAMPLES:

            Original instruction:
            {instruction_example}

            Example verilog:
            {verilog_code_example}
            ''',
        ("verilog", "instruction"): f'''
            EXAMPLES:

            Original verilog:
            {verilog_code_example}

            Example instruction:
            {instruction_example}
            ''',
        ("description", "verilog"): f'''
            EXAMPLES:

            Original description:
            {description_example}

            Example verilog:
            {verilog_code_example}
            ''',
        ("verilog", "description"): f'''
            EXAMPLES:

            Original verilog:
            {verilog_code_example}

            Example description:
            {description_example}
            ''',
    }
    example = examples.get((input_type, output_type), "")

    # Construct the system message
    system_message = f'''
TASK: {task}

INSTRUCTIONS:
1. Read the provided {input_type.replace('_', ' ')} carefully.
2. Transform it into a {output_type.replace('_', ' ')}.
3. Maintain all original details without adding or changing any information.
4. Use a neutral, descriptive tone.

IMPORTANT:
- Do not include any prefixes (e.g., "Here is the rewritten {output_type.replace('_', ' ')}:").
- Provide only the {output_type.replace('_', ' ')}, with no additional text.
- Ensure all technical details from the original {input_type.replace('_', ' ')} are preserved.

{example}
'''

    return system_message.strip()

def generate_user_message(input_type, output_type):
    # Validate input and output types
    valid_types = ["instruction", "description", "verilog"]
    if input_type not in valid_types or output_type not in valid_types:
        raise ValueError(f"input_type and output_type must be one of {valid_types}")

    # Define user message based on input and output types
    task_mapping = {
        ("instruction", "description"): "Rewrite the following instruction as a description:",
        ("description", "instruction"): "Rewrite the following description into a clear and concise instruction:",
        ("instruction", "verilog"): "Develop functional verilog code for the following instruction:",
        ("verilog", "instruction"): "Generate a design instruction based on the provided verilog code:",
        ("description", "verilog"): "Develop functional verilog code based on the provided description:",
        ("verilog", "description"): "Create a descriptive statement that explains the functionality of the provided verilog code:"
    }

    task = task_mapping.get((input_type, output_type))
    if not task:
        raise ValueError(f"Transformation from {input_type} to {output_type} is not supported.")

    return task

def main():
    logger.info(f"Starting main function with input file: {args.json_file}, and max batch size: {args.max_batch_size}")

    system_message = generate_system_message(args.input_field, args.output_field)
    user_message = generate_user_message(args.input_field, args.output_field)
    llm = LLM(model=args.model, tensor_parallel_size=args.num_gpus, enable_prefix_caching=True, max_model_len=8192) #,enable_prefix_caching=True) # max_model_len is a supported parameter
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=42, max_tokens=args.max_tokens)
    new_json_path = args.json_file.replace(".json", f"_{args.model.split('/')[-1]}_with_{args.output_field}.json")
    
    # Create a new empty JSON file if it doesn't exist
    with open(new_json_path, 'w') as f:
        pass  

    with open(args.json_file) as f1:
        json_content = json.load(f1)

    prompts = []
    request_ids = []
    total_items = len(json_content.keys())
    processed_items = 0
    batch_count = 0

    for i in json_content.keys():
        try:
            rtl_id = i
            rtl_input = json_content[i][args.input_field]
            prompts.append(create_prompt(rtl_input, system_message, user_message))
            request_ids.append(rtl_id)
            processed_items += 1
        except KeyError as e:
            logger.warning(f"KeyError for item {i}: {str(e)}")
            logger.warning(f"Content of json_content[{i}]: {json_content[i]}")

        if len(prompts) == args.max_batch_size or processed_items == total_items:
            batch_count += 1
            logger.info(f"Processing batch {batch_count}")
            completed_requests = process_batch(prompts, request_ids, llm, sampling_params)
            if completed_requests:
                join_requests(json_content, completed_requests, new_json_path, args.input_field, args.output_field)
            remaining_batches = (total_items - processed_items + len(prompts) - 1) // args.max_batch_size
            logger.info(f"Batch {batch_count} completed. Remaining batches: {remaining_batches}")
            prompts = []
            request_ids = []

    logger.info(f"Processing completed successfully. Output written to {new_json_path}")

if __name__ == "__main__":
    main()
