#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"

using namespace llvm;

static cl::opt<std::string> TargetFileName(
    "filename",
    cl::desc("Source file name"),
    cl::value_desc("string")
    // cl::init("a.c")
);
static cl::opt<unsigned> TargetLineNumber(
    "lineno",
    cl::desc("Line number of the function"),
    cl::value_desc("unsigned integer")
    // cl::init(0)
);
static cl::opt<std::string> VarName(
    "variable-name",
    cl::desc("Name of variable"),
    cl::value_desc("string")
    // cl::init("a.c")
);

namespace {
  bool inTargetFile(const Function &F, StringRef TargetFileName) {
      if (auto *SP = F.getSubprogram()) {
        if (auto *File = SP->getFile()) {
          StringRef FileName = File->getFilename();
          if (FileName.endswith(TargetFileName) || TargetFileName.endswith(FileName)) {
            //errs() << "Found file " << FileName << "\n";
            return true;
          }
        }
      }
      return false;
  }

  BasicBlock *findTargetBasicBlock(Function &F, unsigned LineNum) {
      if (auto *SP = F.getSubprogram()) {
        if (SP->getLine() == LineNum) {
          //errs() << "Found line " << LineNum << "\n";
          return &F.front();
        }
      }

      for (auto &BB : F) {
        for (auto &I : BB) {
          if (auto DL = I.getDebugLoc()) {
            if (DL.getLine() == LineNum) {
              //errs() << "Found line " << LineNum << "\n";
              return &BB;
            }
          }
        }
      }
      return nullptr;
  }

  struct VariableDefinitionFinderPass : public PassInfoMixin<VariableDefinitionFinderPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
      if (!inTargetFile(F, TargetFileName) || !findTargetBasicBlock(F, TargetLineNumber)) {
        return PreservedAnalyses::all();
      }
      errs() << "Found function " << F.getName() << " on line " << TargetLineNumber << " in file " << TargetFileName << "\n";

      Value *TargetAlloca = nullptr; // This will store the alloca for our variable

      // Step 1: Find the alloca corresponding to the variable via dbg.declare or dbg.value
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (DbgDeclareInst *DbgDeclare = dyn_cast<DbgDeclareInst>(&I)) {
            if (DbgDeclare->getVariable()->getName() == VarName) {
              TargetAlloca = DbgDeclare->getAddress();
              // errs() << "Found variable '" << VarName << "' with DbgDeclare\n";
              goto step2;
            }
          } else if (DbgValueInst *DbgValue = dyn_cast<DbgValueInst>(&I)) {
            if (DbgValue->getVariable()->getName() == VarName) {
              TargetAlloca = DbgValue->getValue();
              // errs() << "Found variable '" << VarName << "' with DbgValue\n";
              goto step2;
            }
          }
        }
      }

      // If the variable wasn't found, exit early
      if (!TargetAlloca) {
        errs() << "Variable '" << VarName << "' not found in debug info.\n";
        return PreservedAnalyses::all();
      }

      // Step 2: Find all stores to that alloca and print debug info
step2:
      for (auto &BB : F) {
        for (auto &I : BB) {
          if (StoreInst *Store = dyn_cast<StoreInst>(&I)) {
            if (Store->getPointerOperand() == TargetAlloca) {
              if (DILocation *Loc = I.getDebugLoc()) {
                unsigned Line = Loc->getLine();
                StringRef File = Loc->getFilename();
                errs() << "Variable '" << VarName << "' defined at line " << Line << " in file " << File << "\n";
              } else {
                // errs() << "Store to variable '" << VarName << "' has no debug location.\n";
              }
            }
          }
        }
      }

      return PreservedAnalyses::all();
    }

    static bool isRequired() { return true; }
  };
}

// Register the pass with LLVM
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "VariableDefinitionFinderPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "variable-def-finder") {
                    FPM.addPass(VariableDefinitionFinderPass());
                    return true;
                  }
                  return false;
                });
          }};
}
