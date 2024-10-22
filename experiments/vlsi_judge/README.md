# vLLM RTL Judge

This script uses the vLLM (very Large Language Model) library to compare RTL (Register-Transfer Level) descriptions, instructions, or Verilog code against a reference. It evaluates the similarity, guesses the design type and name, and provides an overall assessment.

## Features

- Compares two RTL descriptions, instructions, or Verilog code samples against a reference
- Uses vLLM for efficient large language model inference
- Supports various models, including Llama-3.1 and Llama-3.2 series
- Configurable tensor parallelism for multi-GPU setups
- Batch processing for improved performance
- Detailed logging and error handling
- Predefined lists of design types and names for accurate categorization
- Robust JSON parsing with automatic error correction
- Retry mechanism for failed comparisons
- Output review functionality with filtering for low similarity scores and component mismatches

## Main Functions

### `main()`
The entry point of the script. It sets up the LLM, processes the input JSON files in batches, and writes the evaluation results to an output JSON file. It also handles retrying of failed comparisons.

### `generate_system_message(field)`
Generates a system message for the LLM based on the specified field (instruction, description, or verilog).

### `generate_user_message(field)`
Generates a user message based on the field, specifying the comparison task.

### `create_prompt(true_content, content1, content2, system_message, user_message)`
Creates a prompt for the LLM, combining the system message, user message, and the content to be compared.

### `process_batch(prompts, llm, sampling_params)`
Processes a batch of prompts using the LLM and returns the completed requests. It includes error handling and automatic JSON correction for common issues.

### `send_request(prompts, llm, sampling_params)`
Sends a batch request to the LLM and returns the outputs.

### `write_responses(new_data, output_file, index)`
Writes the processed data to the output JSON file, either creating a new file or updating an existing one. It now includes the index for each response.

### `review_output(judged_output, component_mismatch_output_file, low_similarity_output_file, threshold)`
Reviews the output JSON file, filters items with low similarity scores or component mismatches, and saves them to separate files. It also provides a summary of the review process.

## Usage

Run the script with the following command:

```
python vlsi_judge.py [arguments]
```

### Arguments

- `--model`: Model to use for inference (default: "meta-llama/Llama-3.1-8B-Instruct")
- `--json1`: First input JSON file (default: "small_rtl_Llama-3.1-70B-Instruct_with_description.json")
- `--json2`: Second input JSON file (default: "rtl_Llama-3.2-3B-Instruct_with_description.json")
- `--json3`: Reference JSON file (default: "rtl_Llama-3.1-8B-Instruct_with_description.json")
- `--output_file`: Output JSON file (default: "judge_output.json")
- `--max_batch_size`: Maximum batch size for inference (default: 1000)
- `--temperature`: Sampling temperature (default: 0.4)
- `--top_p`: Top-p sampling parameter (default: 0.9)
- `--max_tokens`: Maximum number of tokens to generate (default: 500)
- `--log_file`: Log file path (default: "vllm_judge.log")
- `--num_gpus`: Number of GPUs to use for inference (default: 2)
- `--field`: Field to compare (choices: "instruction", "description", "verilog", default: "description")

## Input/Output

The script reads from three JSON files containing RTL descriptions, instructions, or Verilog code. It compares the contents of the first two files against the reference file (third file) and writes the evaluation results to a new JSON file.

## Logging

The script provides detailed and consistent logging throughout its execution. It logs its progress, any errors, and detailed information about the processing of each prompt. Logs are written to both a file and the console, with error messages being printed to the console for immediate attention.

## Dependencies

- vllm
- argparse
- json
- logging
- re

## Error Handling and Retry Mechanism

The script includes robust error handling for JSON parsing issues. It attempts to automatically fix common JSON errors such as missing commas and unquoted property names. If a comparison fails to process, it is added to a list of skipped items which are retried at the end of the main processing loop.

## Output Format

The script generates a JSON file with the following structure for each comparison:

```json
{
  "index1": {
    "Description1": {
      "Similarity_Score": 8,
      "Design_Type": "Arithmetic Modules",
      "Design_Name": "Adder"
    },
    "Description2": {
      "Similarity_Score": 7,
      "Design_Type": "Arithmetic Modules",
      "Design_Name": "Adder"
    },
    "Overall_Assessment": "Both descriptions are similar to the reference, with Description1 being slightly more accurate..."
  },
  "index2": {
    // ... similar structure for the next comparison
  }
}
```

Each comparison is identified by its original index from the input data, allowing for easy lookup and correlation with the input.

## Output Review

The `review_output` function processes the judge output file and generates two additional files:

1. `low_similarity_items.json`: Contains items with similarity scores below a specified threshold.
2. `component_mismatch_items.json`: Contains items where the Design_Type or Design_Name differs between Description1 and Description2.

This function provides a summary of the review process, including the total number of items, items with low similarity scores, and items with component mismatches.

## Performance and Scalability

The script supports larger batch sizes (default: 1000) and can handle a higher number of comparisons efficiently. The retry mechanism ensures that even if some comparisons fail initially, they are attempted again, maximizing the number of successful evaluations.