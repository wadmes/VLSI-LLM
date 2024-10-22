import json
import logging
import argparse
from typing import List, Dict
from vllm import LLM, SamplingParams
from json import JSONDecodeError
import re

# Set up argument parser
parser = argparse.ArgumentParser(description="Compare RTL fields using vLLM.")
parser.add_argument("--model", default="meta-llama/Llama-3.1-70B-Instruct", help="Model to use for inference")
parser.add_argument("--json1", default="small_rtl_Llama-3.1-70B-Instruct_with_description.json", help="First input JSON file")
parser.add_argument("--json2", default="rtl_Llama-3.2-3B-Instruct_with_description.json", help="Second input JSON file")
parser.add_argument("--json3", default="rtl_Llama-3.1-8B-Instruct_with_description.json", help="Reference JSON file")
parser.add_argument("--output_file", default="judge_output.json", help="Output JSON file")
parser.add_argument("--max_batch_size", type=int, default=1000, help="Maximum batch size for inference")
parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
parser.add_argument("--max_tokens", type=int, default=500, help="Maximum number of tokens to generate")
parser.add_argument("--log_file", default="vllm_judge.log", help="Log file path")
parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for inference")
parser.add_argument("--field", default="description", choices=["instruction", "description", "verilog"], help="Field to compare")
parser.add_argument("--component_mismatch_output_file", default="component_mismatch_items.json", help="Output file for component mismatch items")
parser.add_argument("--low_similarity_output_file", default="low_similarity_items.json", help="Output file for low similarity items")
parser.add_argument("--threshold", type=int, default=7, help="Similarity score threshold")
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

# Define design types and names
DESIGN_TYPES = [
    "Arithmetic Modules",
    "Memory Modules",
    "Control Modules",
    "Miscellaneous Modules"
]

DESIGN_NAMES = [
    # Arithmetic Modules
    "Adder",
    "adder_8bit",
    "adder_16bit",
    "adder_32bit",
    "adder_pipe_64bit",
    "adder_bcd",
    "Substractor",
    "sub_64bit",
    "Multiplier",
    "multi_8bit",
    "multi_16bit",
    "multi_booth_8bit",
    "multi_pipie_4bit",
    "multi_pipie_8bit",
    "Divider",
    "div_16bit",
    "radix2_div",
    "Comparator",
    "comparator_3bit",
    "comparator_4bit",
    "Accumulator",
    "accu",
    "Other Units",
    "fixed_point_adder",
    "fixed_point_substractor",
    "float_multi",
    
    # Memory Modules
    "FIFO (First-In, First-Out)",
    "asyn_fifo",
    "LIFO (Last-In, First-Out)",
    "LIFObuffer",
    "Shifter",
    "right_shifter",
    "LFSR",
    "barrel_shifter",
    
    # Control Modules
    "Finite State Machine (FSM)",
    "fsm",
    "sequence_detector",
    "Counter",
    "counter_12",
    "JC_counter",
    "ring_counter",
    "up_down_counter",
    
    # Miscellaneous Modules
    "Signal generation",
    "signal_generator",
    "square_wave",
    "RISC-V",
    "clkgenerator",
    "instr_reg",
    "ROM",
    "RAM",
    "alu",
    "pe",
    "Frequency divider",
    "freq_div",
    "freq_divbyeven",
    "freq_divbyodd",
    "freq_divbyfrac",
    "Others",
    "calendar",
    "traffic_light",
    "width_8to16",
    "synchronizer",
    "edge_detect",
    "pulse_detect",
    "parallel2serial",
    "serial2parallel"
]

def write_responses(new_data: Dict, output_file: str, index: str):
    logger.info(f"Writing response for index {index} to {output_file}")
    try:
        with open(output_file, 'r+') as file:
            file.seek(0)
            content = file.read().strip()
            if not content:
                logger.debug(f"Output file {output_file} is empty. Writing new data.")
                json.dump({index: new_data}, file, indent=2)
            else:
                logger.debug(f"Updating existing data in {output_file}")
                file.seek(0)
                existing_data = json.load(file)
                existing_data[index] = new_data
                file.seek(0)
                file.truncate()
                json.dump(existing_data, file, indent=2)
        logger.info(f"Response for index {index} written successfully to {output_file}")
    except Exception as e:
        logger.error(f"Error writing index {index} to {output_file}: {str(e)}")

def send_request(prompts: List[List[Dict]], llm: LLM, sampling_params: SamplingParams) -> List:
    logger.info(f"Sending batch request with {len(prompts)} prompts")
    try:
        outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        logger.info(f"Received {len(outputs)} responses")
        return outputs
    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}")
        return []

def create_prompt(true_content: str, content1: str, content2: str, system_message: str, user_message: str) -> List[Dict]:
    return [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"{user_message}\n\nTrue Content: {true_content}\nContent 1: {content1}\nContent 2: {content2}"
        },
    ]

def process_batch(prompts: List[List[Dict]], llm: LLM, sampling_params: SamplingParams) -> List[Dict]:
    logger.info(f"Processing batch of {len(prompts)} prompts")
    outputs = send_request(prompts, llm, sampling_params)
    completed_requests = []
    for i, output in enumerate(outputs):
        try:
            response_text = output.outputs[0].text
            logger.debug(f"Raw output for prompt {i}: {response_text[:100]}...")  # Log first 100 chars
            
            # Try to fix common JSON errors
            response_text = response_text.strip().rstrip(',')
            if not response_text.endswith('}'):
                response_text += '}'
            completed_requests.append(json.loads(response_text))
            logger.debug(f"Successfully parsed JSON for prompt {i}")
        except JSONDecodeError as e:
            logger.error(f"JSON decode error for prompt {i}: {str(e)}")
            logger.debug(f"Problematic output for prompt {i}: {response_text}")
            
            if "Expecting property name enclosed in double quotes" in str(e):
                try:
                    fixed_response = re.sub(r'(\w+)(?=\s*:)', r'"\1"', response_text)
                    completed_requests.append(json.loads(fixed_response))
                    logger.info(f"Fixed JSON by enclosing property names in double quotes for prompt {i}")
                except JSONDecodeError as e2:
                    logger.error(f"Failed to fix JSON with unquoted property names for prompt {i}: {str(e2)}")
                    completed_requests.append(None)
            elif "Expecting ',' delimiter" in str(e):
                try:
                    error_pos = e.pos
                    fixed_response = response_text[:error_pos] + ',' + response_text[error_pos:]
                    fixed_response = re.sub(r',\s*,', ',', fixed_response)
                    completed_requests.append(json.loads(fixed_response))
                    logger.info(f"Fixed JSON by inserting a missing comma for prompt {i}")
                except JSONDecodeError as e2:
                    logger.error(f"Failed to fix JSON with missing comma for prompt {i}: {str(e2)}")
                    completed_requests.append(None)
            else:
                try:
                    fixed_response = response_text.rstrip() + ','
                    completed_requests.append(json.loads(fixed_response))
                    logger.info(f"Fixed JSON by appending a comma for prompt {i}")
                except JSONDecodeError:
                    try:
                        partial_json = json.loads(response_text[:e.pos])
                        completed_requests.append(partial_json)
                        logger.info(f"Salvaged partial JSON for prompt {i}")
                    except:
                        logger.error(f"Failed to salvage partial JSON for prompt {i}")
                        completed_requests.append(None)
        except Exception as e:
            logger.error(f"Failed to process output for prompt {i}: {str(e)}")
            completed_requests.append(None)
    
    logger.info(f"Completed processing batch of {len(prompts)} prompts. Successful: {sum(1 for r in completed_requests if r is not None)}")
    return completed_requests

def generate_system_message(field: str) -> str:
    base_message = f'''
You are a Verilog engineer comparing two {field.replace('_', ' ')}s against a reference {field.replace('_', ' ')}.

INSTRUCTIONS:
1. Read the True {field.replace('_', ' ').capitalize()}, {field.replace('_', ' ').capitalize()} 1, and {field.replace('_', ' ').capitalize()} 2 carefully.
2. Compare {field.replace('_', ' ').capitalize()} 1 and {field.replace('_', ' ').capitalize()} 2 separately to the True {field.replace('_', ' ').capitalize()}.
3. Focus only on the digital logic and functionality described.
4. Assign a similarity score to each {field.replace('_', ' ')} on a scale of 1-10.
5. Guess the design type and design name for both {field.replace('_', ' ').capitalize()} 1 and {field.replace('_', ' ').capitalize()} 2 from the provided lists.
6. Provide your analysis in the exact JSON format specified below.

SCORING SCALE:
- 1 = Completely different, no similarity
- 5 = Somewhat similar, but significant differences
- 10 = Identical in logic and functionality

DESIGN TYPES:
{', '.join(DESIGN_TYPES)}

DESIGN NAMES:
{', '.join(DESIGN_NAMES)}

OUTPUT FORMAT:
Return a JSON object with the following structure:

{{
  "{field.replace('_', ' ').capitalize()}1": {{
    "Similarity_Score": [score from 1-10],
    "Design_Type": "[Your guess of the design type from the provided list]",
    "Design_Name": "[Your guess of the design name from the provided list]"
  }},
  "{field.replace('_', ' ').capitalize()}2": {{
    "Similarity_Score": [score from 1-10],
    "Design_Type": "[Your guess of the design type from the provided list]",
    "Design_Name": "[Your guess of the design name from the provided list]"
  }},
  "Overall_Assessment": "[Brief explanation of your scoring and guesses]"
}}

IMPORTANT:
- Ensure your response is valid JSON, and terminated with a comma.
- Use double quotes for JSON strings, and single quotes for everything else. (i.e. "Overall_Assessment": "...design for a 'square_wave'...")
- Provide a brief explanation in the Overall_Assessment field.
- Choose the Design_Type and Design_Name from the provided lists for both examples.
'''

    if field == "instruction":
        return base_message + '''
ADDITIONAL INSTRUCTIONS:
- Pay attention to the clarity and completeness of the design instructions.
- Consider whether all necessary design requirements are specified.
'''
    elif field == "description":
        return base_message + '''
ADDITIONAL INSTRUCTIONS:
- Focus on the accuracy and completeness of the circuit description.
- Consider whether all key components and their interactions are properly described.
'''
    elif field == "verilog":
        return base_message + '''
ADDITIONAL INSTRUCTIONS:
- Analyze the correctness and efficiency of the verilog code.
- Consider factors such as signal declarations, logic implementation, and potential timing issues.
'''
    else:
        raise ValueError(f"Invalid field: {field}")

def generate_user_message(field: str) -> str:
    return f"Please compare the following {field.replace('_', ' ')}s, provide your analysis, and guess the design type and name:"

def review_output(judged_output: str, component_mismatch_output_file: str, low_similarity_output_file: str, threshold: int) -> None:
    logger.info(f"Starting review of output file: {judged_output}")
    
    try:
        with open(judged_output, 'r') as f:
            json_contents = json.load(f)
        logger.info(f"Successfully loaded {judged_output}")
    except Exception as e:
        logger.error(f"Error loading {judged_output}: {str(e)}")
        return

    # Filter items with similarity score lower than threshold
    low_similarity_items = {}
    component_mismatch = {}
    for i in json_contents.keys():
        print(f"Processing item {i}")
        try:
            if (json_contents[i]["Description1"]["Similarity_Score"] < threshold or
                json_contents[i]["Description2"]["Similarity_Score"] < threshold):
                low_similarity_items[i] = json_contents[i]
            if (json_contents[i]["Description1"]["Design_Type"] != json_contents[i]["Description2"]["Design_Type"] or
                json_contents[i]["Description1"]["Design_Name"] != json_contents[i]["Description2"]["Design_Name"]):
                component_mismatch[i] = json_contents[i]
        except Exception as e:
            logger.error(f"Error processing item {i}: {str(e)}")

    # Save filtered items to output files
    try:
        with open(low_similarity_output_file, 'w') as f:
            json.dump(low_similarity_items, f, indent=2)
        logger.info(f"Saved {len(low_similarity_items)} low similarity items to {low_similarity_output_file}")
    except Exception as e:
        logger.error(f"Error saving low similarity items to {low_similarity_output_file}: {str(e)}")

    try:
        with open(component_mismatch_output_file, 'w') as f:
            json.dump(component_mismatch, f, indent=2)
        logger.info(f"Saved {len(component_mismatch)} component mismatch items to {component_mismatch_output_file}")
    except Exception as e:
        logger.error(f"Error saving component mismatch items to {component_mismatch_output_file}: {str(e)}")

    # Log summary
    logger.info(f"Review summary:")
    logger.info(f"Total items: {len(json_contents)}")
    logger.info(f"Items with similarity score < {threshold}: {len(low_similarity_items)}")
    logger.info(f"Items with component mismatch: {len(component_mismatch)}")

    # Print summary to console
    print(f'Total items: {len(json_contents)}')
    print(f'Items with similarity score < {threshold}: {len(low_similarity_items)}')
    print(f'Saved low similarity items to {low_similarity_output_file}')
    print(f'Items with component mismatch: {len(component_mismatch)}')
    print(f'Saved component mismatch items to {component_mismatch_output_file}')

def main():
    logger.info(f"Starting main function with input files: {args.json1}, {args.json2}, {args.json3}")

    system_message = generate_system_message(args.field)
    user_message = generate_user_message(args.field)
    llm = LLM(model=args.model, tensor_parallel_size=args.num_gpus, max_model_len=8192)
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=42, max_tokens=args.max_tokens)

    # Initialize output file
    with open(args.output_file, 'w') as f:
        json.dump({}, f)
    logger.info(f"Initialized output file: {args.output_file}")

    # Load JSON files
    json_contents = []
    for json_file in [args.json1, args.json2, args.json3]:
        try:
            with open(json_file) as f:
                json_contents.append(json.load(f))
            logger.info(f"Loaded JSON file: {json_file}")
        except Exception as e:
            logger.error(f"Error loading JSON file {json_file}: {str(e)}")

    prompts = []
    total_items = len(json_contents[0].keys())
    processed_items = 0
    batch_count = 0
    skipped_items = []

    logger.info(f"Starting processing of {total_items} items")

    for i in json_contents[0].keys():
        try:
            content1 = json_contents[0][i][args.field]
            content2 = json_contents[1][i][args.field]
            true_content = json_contents[2][i][args.field]
            prompts.append((i, create_prompt(true_content, content1, content2, system_message, user_message)))
            processed_items += 1
            logger.debug(f"Created prompt for item {i}")
        except KeyError as e:
            logger.warning(f"KeyError for item {i}: {str(e)}")

        if len(prompts) == args.max_batch_size or processed_items == total_items:
            batch_count += 1
            logger.info(f"Processing batch {batch_count}")
            completed_requests = process_batch([p[1] for p in prompts], llm, sampling_params)
            for (index, _), response in zip(prompts, completed_requests):
                if response is not None:
                    write_responses(response, args.output_file, index)
                    logger.debug(f"Wrote response for index {index}")
                else:
                    logger.warning(f"Skipping write for index {index} due to processing error")
                    skipped_items.append((index, _))
            remaining_batches = (total_items - processed_items + len(prompts) - 1) // args.max_batch_size
            logger.info(f"Batch {batch_count} completed. Remaining batches: {remaining_batches}")
            prompts = []

    # Retry skipped items
    if skipped_items:
        logger.info(f"Retrying {len(skipped_items)} skipped items")
        retry_prompts = [prompt for _, prompt in skipped_items]
        retry_completed_requests = process_batch(retry_prompts, llm, sampling_params)
        for (index, _), response in zip(skipped_items, retry_completed_requests):
            if response is not None:
                write_responses(response, args.output_file, index)
                logger.info(f"Successfully processed index {index} after retry")
            else:
                logger.error(f"Failed to process index {index} after retry")

    logger.info(f"Processing completed. Total items: {total_items}, Successfully processed: {total_items - len(skipped_items)}, Failed: {len(skipped_items)}")
    review_output(args.output_file, args.component_mismatch_output_file, args.low_similarity_output_file, args.threshold)
if __name__ == "__main__":
    main()
