# LLVM tools

## Build
```
export LLVM_DIR=/usr/lib/llvm-17
cmake -DLT_LLVM_INSTALL_DIR=$LLVM_DIR -S. -Bbuild
cmake --build build
```
