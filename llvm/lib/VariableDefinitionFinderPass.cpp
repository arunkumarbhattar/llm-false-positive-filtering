#include "helper.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <nlohmann/json.hpp>

using namespace llvm;

namespace {
struct VariableDefinitionFinderPass
    : public PassInfoMixin<VariableDefinitionFinderPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM) {
    nlohmann::json ReqJson;
    nlohmann::json RespJson;
    std::string ErrMsg;
    llvm::raw_string_ostream ErrS(ErrMsg);
    raw_ostream &OS = ErrS;
    // raw_ostream &OS = llvm::errs();

    if (!readRequest(ReqJson)) {
      OS << "Error: Could not open named pipe for reading.\n";
      RespJson["error"]["message"] = ErrMsg;
      writeResponse(RespJson);
      return PreservedAnalyses::all();
    }
    std::string TargetFileName = ReqJson["args"]["filename"];
    unsigned TargetLineNumber = ReqJson["args"]["lineno"];
    std::string VarName = ReqJson["args"]["varname"];

    auto *FPtr = findFunction(M, TargetFileName, TargetLineNumber);
    if (!FPtr) {
      OS << "Error: Could not find function on " << TargetFileName << ":"
         << TargetLineNumber << "\n";
      RespJson["error"]["message"] = ErrMsg;
      writeResponse(RespJson);
      return PreservedAnalyses::all();
    }
    auto &F = *FPtr;

    Value *TargetAlloca =
        nullptr; // This will store the alloca for our variable

    // Step 1: Find the alloca corresponding to the variable via dbg.declare or
    // dbg.value
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (DbgDeclareInst *DbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
          if (DbgDeclare->getVariable()->getName() == VarName) {
            TargetAlloca = DbgDeclare->getAddress();
             errs() << "Found variable '" << VarName << "' with DbgDeclare\n";
            goto step2;
          }
        } else if (DbgValueInst *DbgValue = dyn_cast<DbgValueInst>(&I)) {
          if (DbgValue->getVariable()->getName() == VarName) {
            TargetAlloca = DbgValue->getValue();
            errs() << "Found variable '" << VarName << "' with DbgValue\n";
            goto step2;
          }
        }
      }
    }

    // If the variable wasn't found, exit early
    if (!TargetAlloca) {
      OS << "Error: Variable '" << VarName << "' not found in debug info.\n";
      RespJson["error"]["message"] = ErrMsg;
      writeResponse(RespJson);
      return PreservedAnalyses::all();
    }

    // Step 2: Find all stores to that alloca and print debug info
  step2:
    std::vector<unsigned> DefLines;
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
          if (Store->getPointerOperand() == TargetAlloca) {
            if (DILocation *Loc = I.getDebugLoc()) {
              unsigned Line = Loc->getLine();
               StringRef File = Loc->getFilename();
               OS << "Variable '" << VarName << "' defined at line " << Line
               << " in file " << File << "\n";
              DefLines.push_back(Line);
            } else {
               OS << "Store to variable '" << VarName << "' has no debug location.\n";
            }
          }
        }
      }
    }

    RespJson["result"] = DefLines;
    writeResponse(RespJson);
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

// Register the pass with LLVM
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "VariableDefinitionFinderPass",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "variable-def-finder") {
                    MPM.addPass(VariableDefinitionFinderPass());
                    return true;
                  }
                  return false;
                });
          }};
}
