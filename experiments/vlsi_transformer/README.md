# VLSI Transformer

This script processes RTL (Register-Transfer Level) instructions, descriptions, or code using the vLLM (very Large Language Model) library. It can transform between RTL instructions, descriptions, and verilog code in various combinations.

## Main Functions

### `main()`
The entry point of the script. It sets up the LLM, processes the input JSON file in batches, and writes the transformed output to a new JSON file.

### `generate_system_message(input_type, output_type)`
Generates a system message for the LLM based on the specified input and output types. This function defines the task and provides examples for the model.

### `generate_user_message(input_type, output_type)`
Generates a user message based on the input and output types, specifying the transformation task.

### `create_prompt(input_text, system_message, user_message)`
Creates a prompt for the LLM, combining the system message, user message, and the input text.

### `process_batch(prompts, request_ids, llm, sampling_params)`
Processes a batch of prompts using the LLM and returns the completed requests.

### `send_request(prompts, llm, sampling_params)`
Sends a batch request to the LLM and returns the outputs.

### `join_requests(json_content, completed_requests, json_file, input_field, output_field)`
Updates the JSON content with the completed requests and writes the results to the output file.

### `write_responses(new_data, output_file)`
Writes the processed data to the output JSON file, either creating a new file or updating an existing one.

## Usage

Run the script with the following command:
python vlsi_transformer.py [arguments]

### Arguments

- `--model`: Model to use for inference (default: "meta-llama/Llama-3.1-8B-Instruct")
- `--json_file`: Input JSON file (default: "rtl.json")
- `--max_batch_size`: Maximum batch size for inference (default: 1000)
- `--temperature`: Sampling temperature (default: 0.4)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--max_tokens`: Maximum number of tokens to generate (default: 200)
- `--log_file`: Log file path (default: "vllm_processing.log")
- `--num_gpus`: Number of GPUs to use for inference (default: 1)
- `--input_field`: Input field in JSON to convert from (default: "instruction")
- `--output_field`: Output field in JSON to convert to (default: "description")

## Input/Output

The script reads from a JSON file containing RTL instructions, descriptions, or verilog code and writes the transformed output to a new JSON file. The output file name is based on the input file name, the model used, and the output field type.

## Logging

The script logs its progress and any errors to both a file and the console. Detailed logs are written to the specified log file, while error messages are also printed to the console.

## Dependencies

- vllm
- argparse
- json
- logging

## Usage Examples - Adjust the arguments as needed

1. Convert RTL instructions to descriptions:
python vlsi_transformer.py --input_field instruction --output_field description

2. Generate verilog code from instructions:
python vlsi_transformer.py --input_field instruction --output_field verilog --max_tokens 500

3. Create RTL instructions from verilog code:
python vlsi_transformer.py --input_field verilog --output_field instruction --model meta-llama/Llama-3.1-70B-Instruct --num_gpus 4

4. Process a custom JSON file with a different model:
python vlsi_transformer.py --json_file custom_rtl.json --model meta-llama/Llama-3.1-70B-Instruct --input_field description --output_field verilog --num_gpus 4

5. Adjust generation parameters:
python vlsi_transformer.py --temperature 0.7 --top_p 0.95 --max_tokens 300

## Performance Metrics

I conducted performance tests using various configurations of vLLM with different models and tensor parallelism settings. The results are summarized below:

### vLLM Tensor Parallelism 4x

Using 4 H100 GPUs, I tested the following models:

- Llama-3.1-70B: 1000 prompts @ 01:42, 9.71 it/s, input speed: 7770.01 tokens/s, output speed: 1230.00 tokens/s
- Llama-3.1-8B: 1000 prompts @ 00:41, 24.36 it/s, input speed: 19485.45 tokens/s, output speed: 2999.42 tokens/s

### vLLM Tensor Parallelism 4x (with max model size of 8192):

- Llama-3.1-70B: 1000 prompts @ 01:07, 14.84 it/s, input speed: 7000.57 tokens/s, output speed: 1572.00 tokens/s
- Llama-3.1-70B (enabling prefix caching): 1000 prompts @ 00:51, 19.43 it/s, est. speed input: 11210.49 tokens/s, output: 2479.68 tokens/s
- Llama-3.1-8B: 1000 prompts @ 00:18, 54.37 it/s, input speed: 25488.83 tokens/s, output speed: 5953.57 tokens/s
- Llama-3.2-3B: 1000 prompts @ 00:16, 61.49 it/s, input speed: 28909.23 tokens/s, output speed: 6864.33 tokens/s

### vLLM Tensor Parallelism 2x (with max model size of 8192)

Using 2 H100 GPUs, I tested the following models:

- Llama-3.1-70B: 1000 prompts @ 03:14, 5.14 it/s, input speed: 4112.01 tokens/s, output speed: 566.21 tokens/s (encountered preemptive handling, equivalent to swap with GPU)
- Llama-3.1-8B: 1000 prompts @ 00:28, 34.72 it/s, input speed: 27771.51 tokens/s, output speed: 4277.88 tokens/s
- Llama-3.2-3B: 1000 prompts @ 00:20, 49.77 it/s, input speed: 29057.41 tokens/s, output speed: 5982.28 tokens/s

### vLLM Tensor Parallelism 2x (with full model size)

Using 2 H100 GPUs, I tested the following models:

- Llama-3.1-70B: Not enough memory
- Llama-3.1-8B: 1000 prompts @ 00:34, 29.24 it/s, input speed: 16869.06 tokens/s, output speed: 3617.33 tokens/s
- Llama-3.2-3B: 1000 prompts @ 00:29, 33.47 it/s, input speed: 19038.32 tokens/s, output speed: 3948.44 tokens/s

### vLLM Tensor Parallelism 1x (with max model size of 8192)

Using 1 H100 GPU, I tested the following models:

- Llama-3.1-70B: Not enough memory
- Llama-3.1-8B: 1000 prompts @ 00:33, 29.96 it/s, input speed: 17040.96 tokens/s, output speed: 3701.92 tokens/s
- Llama-3.2-3B: 1000 prompts @ 00:21, 45.98 it/s, input speed: 26155.70 tokens/s, output speed: 5429.78 tokens/s

### vLLM Tensor Parallelism 1x (with full model size)

Using 1 H100 GPU, I tested the following models:

- Llama-3.1-70B: Not enough memory
- Llama-3.1-8B: 1000 prompts @ 00:35, 28.10 it/s, input speed: 15987.19 tokens/s, output speed: 3444.93 tokens/s
- Llama-3.2-3B: 1000 prompts @ 00:28, 34.80 it/s, input speed: 20175.61 tokens/s, output speed: 4229.47 tokens/s

Future tuning:
- Enable prefix caching
- Tune num_batched_tokens
