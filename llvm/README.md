# LLVM tools

## Build
```
export LLVM_DIR=/usr/lib/llvm-17
cmake -DLT_LLVM_INSTALL_DIR=$LLVM_DIR -S. -Bbuild
cmake --build build
```

## Add LLVM pass
1. Create `lib/foo.cpp`
2. Edit LLVM_TOOLS_PLUGINS and add FOO_SOURCES in `lib/CMakeLists.txt`.
