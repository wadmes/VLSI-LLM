#!/home/crellis/miniconda3/envs/vllm/bin/python
import json
import logging
import argparse
from typing import List, Dict
from vllm import LLM
from vllm.sampling_params import SamplingParams, GuidedDecodingParams
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description="Transform RTL fields using vLLM.")
parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct", help="Model to use for inference")
parser.add_argument("--json_file", default="4k_context.json", help="Input JSON file")
parser.add_argument("--prediction", default="both", help="Prediction to be ran, 'type', 'description', or 'both', default is 'both'")
parser.add_argument("--max_batch_size", type=int, default=30000, help="Max Batch size for batch inference, not implimented by default")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for description prediction")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter for description prediction")
parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens to generate for description prediction")
parser.add_argument("--log_file", default="verilog_transformer.log", help="Log file path")
parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use for inference")
parser.add_argument("--model_len", type=int, default=128000, help="model_len for inilizing model, fits to max model len of 128K. WARNINGL truncate_len must be lowered if this is lowered")
parser.add_argument("--input_field", default="verilog", help="Input field in JSON to convert from [verilog, rtl, or custom field]")
parser.add_argument("--truncate_len", type=int, default=230000, help="length to truncate input to, fits to max model len of 128K")

args = parser.parse_args()

DESIGN_TYPES = ["Encryption Units","Data Path Units","Control Logic Units","Arithmetic Units","Communication Protocol Units","Signal Processing Units","Clock Management Units","Other Units"]


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
    """
    Write or update responses to a JSON file.
    
    Args:
        new_data (Dict): New data to write or update in the file
        output_file (str): Path to the output JSON file
    """
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

def send_request(prompts: List[List[Dict]], llm: LLM) -> List:
    """
    Send batch requests to LLM for description prediction.
    
    Args:
        prompts (List[List[Dict]]): List of prompts to process
        llm (LLM): Initialized LLM model instance
    
    Returns:
        List: Model outputs for each prompt
    """
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, seed=42, max_tokens=args.max_tokens, min_p=0.02, repetition_penalty=1.07)
    logger.info(f"Sending batch request with {len(prompts)} prompts")
    try:
        outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        logger.info(f"Received {len(outputs)} responses")
        return outputs
    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}")
        return []


def create_description_prompt(input_text: str, truncation: int) -> List[Dict]:
    """
    Create a prompt for description generation task.
    
    Args:
        input_text (str): Verilog code to analyze
        truncation (int): Maximum length to truncate input text
    
    Returns:
        List[Dict]: Formatted prompt for the LLM
    """
    if truncation != 0:
        if len(input_text) > truncation:
            input_text = input_text[:truncation]
    return [
        {
            "role": "system",
            "content": '''You are a hardware description expert. Provide a single, coherent technical paragraph describing the functionality of a Verilog module.

Constraints:
- Use complete English sentences.
- Avoid mentioning variable names or including any Verilog syntax.
- Ensure the description focuses on functionality, not implementation details.
- Do not use lists, bullet points, or code snippets.
- Maintain a logical flow without line breaks or special formatting.
            
Example:
---
**Module Description:**
This module implements an edge detection mechanism. It accepts an 8-bit binary input and a clock signal, producing an 8-bit output that reflects the input value one cycle after an edge is detected. The circuit operates by comparing the current input with the previous input to identify edges, utilizing a counter to manage the delay in output generation.
---
'''
        },
        {
            "role": "user",
            "content": f"Provide a detailed description of the following Verilog module:\n{input_text}"
        },
    ]

def process_batch(prompts: List[List[Dict]], request_ids: List[str], llm: LLM) -> Dict:
    """
    Process a batch of description prediction requests.
    
    Args:
        prompts (List[List[Dict]]): List of prompts to process
        request_ids (List[str]): Corresponding IDs for each prompt
        llm (LLM): Initialized LLM model instance
    
    Returns:
        Dict: Mapping of request IDs to their generated descriptions
    """
    logger.info(f"Processing batch of {len(prompts)} prompts")
    outputs = send_request(prompts, llm)
    completed_requests = {}
    for output, request_id in zip(outputs, request_ids):
        try:
            response_text = output.outputs[0].text
            completed_requests[request_id] = {"description": response_text}
        except Exception as e:
            logger.error(f"Failed to process output for request ID {request_id}: {str(e)}")
    logger.info(f"Completed processing batch of {len(prompts)} prompts")
    return completed_requests

def send_labels_request(prompts: List[List[Dict]], llm: LLM, sampling_params: SamplingParams) -> List:
    """
    Send batch requests to LLM for type prediction with guided decoding.
    
    Args:
        prompts (List[List[Dict]]): List of prompts to process
        llm (LLM): Initialized LLM model instance
        sampling_params (SamplingParams): Parameters for guided decoding
    
    Returns:
        List: Model outputs for each prompt
    """
    logger.info(f"Sending batch request with {len(prompts)} prompts")
    try:
        outputs = llm.chat(prompts, sampling_params, use_tqdm=True)
        logger.info(f"Received {len(outputs)} responses")
        return outputs
    except Exception as e:
        logger.error(f"Error in batch inference: {str(e)}")
        return []

def generate_system_message() -> str:
    """
    Generate the system message for type prediction task.
    
    Returns:
        str: Formatted system message with category descriptions
    """
    system_message = f'''ROLE: You are a specialized Verilog code analyzer focused on classifying hardware designs into specific categories. 
    TASK: Analyze Verilog code header and classify it into one of the following categories based on its inputs, outputs, logic, and overall functional behavior.

Possible categories:
{DESIGN_TYPES}

Here is a description of each category:
Encryption Units: Designs implementing cryptographic algorithms, secure hash functions, or other security-related operations
Data Path Units: Components handling data flow, multiplexers, decoders, registers, and data routing
Control Logic Units: State machines, sequence controllers, and decision-making logic
Arithmetic Units: Mathematical operations, ALUs, multipliers, dividers, and computational blocks
Communication Protocol Units: Implementations of protocols like UART, I2C, SPI, or other communication interfaces
Signal Processing Units: Filters, FFT implementations, signal conditioning, and digital signal processing
Clock Management Units: Clock generators, PLL implementations, clock dividers, and timing control
Other Units: Designs that don't clearly fit into the above categories

Respond ONLY with the most appropriate category name from the list above.
'''
    return system_message

def generate_user_message():
    task = f'''Please analyze the following Verilog code and classify it into one of the specified design types. Verilog code to analyze: 
    '''
    return task

def process_labels_batch(prompts: List[List[Dict]], request_ids: List[str], llm: LLM) -> Dict:
    """
    Process a batch of type prediction requests with guided decoding.
    
    Args:
        prompts (List[List[Dict]]): List of prompts to process
        request_ids (List[str]): Corresponding IDs for each prompt
        llm (LLM): Initialized LLM model instance
    
    Returns:
        Dict: Mapping of request IDs to their predicted types
    """
    logger.info(f"Processing batch of {len(prompts)} prompts")
    #sampling parameters found best by trial and error for guided decoding
    sampling_params = SamplingParams(
        seed=42,
        max_tokens=20,
        guided_decoding=GuidedDecodingParams(choice=DESIGN_TYPES),
        temperature=0.4,
        top_p=0.9
    )
    outputs = send_labels_request(prompts, llm, sampling_params)
    completed_requests = {}
    
    for output, request_id in zip(outputs, request_ids):
        raw_text = output.outputs[0].text.strip()
        try:
            response_text = {"function_label": raw_text}
            completed_requests[request_id] = response_text
        except Exception as e:
            logger.warning(f"Failed to format response for {request_id}: {str(e)}")
            completed_requests[request_id] = {"function_label": raw_text}
    
    logger.info(f"Completed processing batch. Successful: {len(completed_requests)} out of {len(prompts)}")
    return completed_requests

def create_labels_prompt(input: str, truncation: int) -> List[Dict]:  
    system_message = generate_system_message()
    user_message = generate_user_message()

    if truncation != 0:
        if len(input) > truncation:
            input = input[:truncation]
    return [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": f"{user_message} {input}"
        },
    ]


def init_llm(model_len: int, model: str) -> LLM:
    """
    Initialize the LLM model with specified parameters.
    
    Args:
        model_len (int): Maximum model context length
        model (str): Model identifier/path
    
    Returns:
        LLM: Initialized LLM instance
    """
    llm = LLM(model=model, tensor_parallel_size=args.num_gpus, max_model_len=model_len, gpu_memory_utilization=0.9)
    return llm

def run_type_prediction(llm: LLM, json_content: Dict, new_json_path: str, truncation = 0):
    """
    Run type prediction on a batch of Verilog code samples.
    
    Args:
        llm (LLM): Initialized LLM model instance
        json_content (Dict): Input data containing Verilog code
        new_json_path (str): Path to save results
        truncation (int, optional): Maximum input length. Defaults to 0 (no truncation)
    """
    logger.info(f"Starting Type Prediction")
    if truncation != 0:
        logger.info(f"Prompt input trucation set to: {truncation}")
    prompts = []
    request_ids = []
    for i in json_content.keys():
        try:
            rtl_id = i
            rtl_input = json_content[i][args.input_field]
            prompts.append(create_labels_prompt(rtl_input, truncation))
            request_ids.append(rtl_id)
        except KeyError as e:
            logger.warning(f"KeyError for item {i}: {str(e)}")
            logger.warning(f"Content of json_content[{i}]: {json_content[i]}")
    # If the dataset length is large, consider using a batch inference method.
    # Impliment using the following function WITHIN the for loop:
        # if len(prompts) == args.batch_size:
        #     completed_requests = process_batch(prompts, request_ids, llm)
    if len(prompts)>0:
        completed_labels = process_labels_batch(prompts, request_ids, llm)
    if completed_labels:
        write_responses(completed_labels, new_json_path)
        logger.info(f"Type Prediction completed, {len(completed_labels)} responses written to {new_json_path}")

def run_description_prediction(llm: LLM, json_content: Dict, new_json_path: str, truncation = 0):
    """
    Run description prediction on a batch of Verilog code samples.
    
    Args:
        llm (LLM): Initialized LLM model instance
        json_content (Dict): Input data containing Verilog code
        new_json_path (str): Path to save results
        truncation (int, optional): Maximum input length. Defaults to 0 (no truncation)
    """
    logger.info(f"Starting Description Prediction")
    if truncation != 0:
        logger.info(f"Prompt input trucation set to: {truncation}")
    prompts = []
    request_ids = []
    for i in json_content.keys():
        try:
            rtl_id = i
            rtl_input = json_content[i][args.input_field]
            prompts.append(create_description_prompt(rtl_input, truncation))
            request_ids.append(rtl_id)
        except KeyError as e:
            logger.warning(f"KeyError for item {i}: {str(e)}")
            logger.warning(f"Content of json_content[{i}]: {json_content[i]}")
    # If the dataset length is large, consider using a batch inference method.
    # Impliment using the following function WITHIN the for loop:
        # if len(prompts) == args.batch_size:
        #     completed_requests = process_batch(prompts, request_ids, llm)
    
    if len(prompts)>0:
        completed_requests = process_batch(prompts, request_ids, llm)
    if completed_requests:
        write_responses(completed_requests, new_json_path)
        logger.info(f"Description prediction completed, {len(completed_requests)} responses written to {new_json_path}")

def main():
    """
    Main execution function that orchestrates the prediction pipeline.
    Handles both type and description prediction based on command line arguments.
    """
    if args.prediction not in ["type", "description", "both"]:
        logger.error(f"Invalid prediction type: {args.prediction}, please choose from 'type', 'description', or 'both'")
        return
    logger.info(f"Starting main function for {args.prediction} prediction experiment(s)")

    with open(args.json_file, 'r') as f:
        json_content = json.load(f)
    llm = init_llm(args.model_len, args.model)

    new_json_path = args.json_file.replace(".json", f"_{args.model.split('/')[-1]}_{args.truncate_len}_{args.input_field}.json")
    with open(new_json_path, 'w') as f:
        pass 
    if args.prediction == "type":
        run_type_prediction(llm, json_content, new_json_path, args.truncate_len)
    elif args.prediction == "description":
        run_description_prediction(llm, json_content, new_json_path, args.truncate_len)
    else:
        run_type_prediction(llm, json_content, new_json_path, args.truncate_len)
        run_description_prediction(llm, json_content, new_json_path, args.truncate_len)
    logger.info(f"Exiting program")

if __name__ == "__main__":
    main()
