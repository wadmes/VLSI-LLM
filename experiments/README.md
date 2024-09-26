# Experiment directory for VLSI-LLM
This directory contains the code to construct the evaluation flow.

## Experiment Flow
The experiment flow is constructed as follows:
1. We first use our graph-LLAMA3 model to describe the VLSI design.
2. We then use the original LLAMA3 model to describe the VLSI design.
3. We ask LLAMA3 to compare the results of the two models