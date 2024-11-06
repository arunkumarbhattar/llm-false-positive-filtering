# LLM-SAST-Triage

## Step 1: Compile LLVM Passes
Refer to `llvm/README.md` for detailed instructions on how to compile the necessary LLVM passes.
Then set the environment variable
``` bash
export LLVM_PASSES_LIB_DIR=/path/to/llm-sast-triage/llvm/build/lib
```

## Step 2: Run the LLM
To analyze warnings in a repository, provide the following inputs:
- Path to the repository
- CodeQL SARIF file
- LLVM bitcode file

### Usage:
```bash
python3 llm_triage.py <repo_path> <sarif_path> <bitcode_path>
```
