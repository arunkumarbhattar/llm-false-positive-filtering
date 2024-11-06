#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include <nlohmann/json.hpp>

bool inTargetFile(const llvm::Function &F, llvm::StringRef TargetFileName);
llvm::BasicBlock *findTargetBasicBlock(llvm::Function &F, unsigned LineNum);
llvm::Function *findFunction(llvm::Module &M, llvm::StringRef FileName,
                             unsigned LineNum);
bool readRequest(nlohmann::json &J);
bool writeResponse(const nlohmann::json &J);
