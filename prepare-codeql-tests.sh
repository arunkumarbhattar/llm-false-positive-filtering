#!/bin/bash

#set -x

# Check if a directory argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <directory>"
  exit 1
fi

DIR=$1
CODEQL=$HOME/codeql-home/codeql/codeql
CODEQL_DB=/tmp/codeql-db
MANUALLY_LABEL_CODEQL="$PWD/manually_label_codeql.py"

# Change to the specified directory
cd "$DIR" || { echo "Failed to change to directory $DIR"; exit 1; }

# Gather all source files that match test*.c or test*.cpp
TESTSRC=$(find . -type f \( -name "test*.c" -o -name "test*.cpp" \))

if [ -z "$TESTSRC" ]; then
  echo "No matching source files found in the directory."
  exit 1
fi

# Create the CodeQL database
$CODEQL database create "$CODEQL_DB" --overwrite --language=cpp --command "gcc -c $TESTSRC"

# Find all *.qlref files in the directory and prepend "codeql/cpp-queries:"
QUERIES=$(find . -type f -name "*.qlref" -exec sed 's_^_codeql/cpp-queries:_' {} +)

if [ -z "$QUERIES" ]; then
  echo "No .qlref files found."
  exit 1
fi

# Run the CodeQL analysis
$CODEQL database analyze "$CODEQL_DB" --format=sarifv2.1.0 --output=cpp.sarif --threads=0 $QUERIES

# Manually label
python3 "$MANUALLY_LABEL_CODEQL" "$DIR"

# Compile .c/.cpp files to .bc files
$LLVM_DIR/bin/clang -c -emit-llvm -O0 -g -Wno-everything $TESTSRC

# Create a variable with the corresponding .bc files
BCFILES=$(echo "$TESTSRC" | sed 's/\.[^.]*$/.bc/g')

# Link all .bc files into one combined file
$LLVM_DIR/bin/llvm-link -o test-combined.bc $BCFILES

echo "All tasks completed successfully."
