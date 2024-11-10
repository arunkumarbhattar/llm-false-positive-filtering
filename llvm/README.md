# LLVM Backend Tools (Facts)

## Overview

This repository contains a collection of LLVM backend tools designed to analyze C/C++ programs at the Intermediate Representation (IR) level. 
These tools leverage LLVM's infrastructure to perform tasks such as extracting control conditions, finding variable definitions, and building control dependence graphs. 
They are particularly useful for static analysis workflows.
## Build
```
export LLVM_DIR=/usr/lib/llvm-17
cmake -DLT_LLVM_INSTALL_DIR=$LLVM_DIR -S. -Bbuild
cmake --build build
```

## Prerequisites

```aiignore
sudo apt-get install llvm-17 llvm-17-dev clang-17
sudo apt-get install cmake
sudo apt-get install nlohmann-json3-dev
sudo apt-get install clang
sudo apt-get install build-essential

```



## LLVM Tools Overview

### VariableDefinitionFinderPass

#### Purpose:
The VariableDefinitionFinderPass is an LLVM pass that identifies all definition points (i.e., store instructions) of a specified variable within a function. It utilizes debug information to accurately locate variable declarations and tracks where the variable is assigned throughout the function.

### ControlDepGraph

#### Purpose:
The ControlDepGraph pass constructs a Control Dependence Graph (CDG) for functions within a module. A CDG represents the dependencies between basic blocks based on control flow, which is essential for understanding complex control structures and potential vulnerabilities.

## Usage Instructions

### VariableDefinitionFinderPass Usage

#### Purpose:
Identify all definition points of a specified variable within a function and output the line numbers where the variable is defined.

```aiignore
Steps:

    Prepare a JSON Request:

    Create a JSON file (e.g., request.json) with the following structure:

{
"args": {
"filename": "path/to/source_file.cpp",
"lineno": 42,
"varname": "myVariable"
}
}

    filename: Path to the source file.
    lineno: Line number where the function containing the variable is located.
    varname: Name of the variable to find definitions for.

```

Run the Pass with opt:

```aiignore
Use the opt tool to apply the pass to your LLVM IR file (input.bc):

opt -load-pass-plugin=./build/libVariableDefinitionFinderPass.so -passes=variable-def-finder input.bc -o output.bc

    -load-pass-plugin: Specifies the path to the compiled pass plugin.
    -passes=variable-def-finder: Invokes the VariableDefinitionFinderPass.
    input.bc: Input LLVM IR file.
    output.bc: Output LLVM IR file after pass execution.
```

Receive the JSON Response:

```aiignore
The pass writes a JSON response containing the line numbers where the variable is defined. 

{
  "result": [45, 50, 60]
}

This indicates that the variable myVariable is defined at lines 45, 50, and 60 in source_file.cpp.
```

similarly there are others -->