#include "helper.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>

// #define MYDEBUG

using namespace llvm;

std::string getControlCondAtLoc(std::string filename, unsigned lineNumber,
                                unsigned columnNumber);

namespace {

std::string getControlTransferCond(BasicBlock *BB, unsigned SuccIdx) {
  auto *Terminator = BB->getTerminator();
  std::string Cond;

  if (auto *BI = dyn_cast<BranchInst>(Terminator)) {
    if (BI->isConditional()) {
      if (auto *CondInst = dyn_cast<Instruction>(BI->getCondition())) {
        if (const DebugLoc &DL = CondInst->getDebugLoc()) {
          // LineNum = DL.getLine();
          //  Use the -> operator for direct access to DILocation methods
          std::string FullFilename =
              (DL->getDirectory() + "/" + DL->getFilename()).str();
          Cond = getControlCondAtLoc(FullFilename, DL.getLine(), DL.getCol());
#ifdef MYDEBUG
          dbgs() << *DL << "\n";
#endif
        }
      }
      // SuccIdx == 0 means BRANCH IS TAKEN
      // SuccIdx == 1 means BRANCH IS NOT TAKEN
      assert(SuccIdx == 0 || SuccIdx == 1);
#ifdef MYDEBUG
      // dbgs() << Cond << " SuccIdx: " << SuccIdx << "\n";
#endif
      if (SuccIdx == 0) {
        return "(" + Cond + ")";
      } else {
        return "!(" + Cond + ")";
      }
    }
  } else if (auto *SI = dyn_cast<SwitchInst>(Terminator)) {
    if (auto *CondInst = dyn_cast<Instruction>(SI->getCondition())) {
      if (const DebugLoc &DL = CondInst->getDebugLoc()) {
        std::string FullFilename =
            (DL->getDirectory() + "/" + DL->getFilename()).str();
        Cond = getControlCondAtLoc(FullFilename, DL.getLine(), DL.getCol());
      }
    }
    if (SuccIdx == 0) {
      std::string Result;
      for (const auto &Case : SI->cases()) {
        if (ConstantInt *CaseValue = Case.getCaseValue()) {
          int64_t caseIntValue = CaseValue->getSExtValue();
          if (!Result.empty()) {
            Result += " /\\ ";
          }
          Result += "(" + Cond + " != " + std::to_string(caseIntValue) + ")";
        }
      }
      return Result;

    } else if (ConstantInt *CaseValue =
                   SI->findCaseDest(SI->getSuccessor(SuccIdx))) {
      // OS << "Switch condition on line " << LineNum << " is equal to "
      //   << CaseValue->getValue() << "\n";
      return "(" + Cond + " == " + std::to_string(CaseValue->getSExtValue()) +
             ")";
    }
  }
  return Cond;
}

struct ControlDepGraph : public AnalysisInfoMixin<ControlDepGraph> {
  // Credit:
  // https://github.com/ARISTODE/program-dependence-graph/blob/main/src/ControlDependencyGraph.cpp
  using Result =
      std::map<BasicBlock *, std::set<std::pair<BasicBlock *, unsigned>>>;
  // Edge A --idx-> B means A depends on B via B's idx-th successor

  Result run(Function &F, FunctionAnalysisManager &FAM) {
    Result CDG;
    auto &PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);

    for (auto &BB : F) {
      if (auto *Terminator = BB.getTerminator()) {
        unsigned NumSuccessors = Terminator->getNumSuccessors();
        // for (auto *Succ : successors(&BB)) {
        for (unsigned SuccIdx = 0; SuccIdx < NumSuccessors; ++SuccIdx) {
          auto *Succ = Terminator->getSuccessor(SuccIdx);
          if (&BB == Succ || !PDT.dominates(Succ, &BB)) {
            auto *NearestCommonDominator =
                PDT.findNearestCommonDominator(&BB, Succ);
            if (NearestCommonDominator == &BB) {
              // BB controls Succ through BB's SuccIdx-th successor
              // CDG[&BB].insert(Succ);
              CDG[Succ].insert({&BB, SuccIdx});
            }
            for (auto *Cur = PDT.getNode(Succ);
                 Cur != PDT.getNode(NearestCommonDominator);
                 Cur = Cur->getIDom()) {
              // BB controls Cur through BB's SuccIdx-th successor
              // CDG[&BB].insert(Cur->getBlock());
              CDG[Cur->getBlock()].insert({&BB, SuccIdx});
            }
          }
        }
      }
    }

    return CDG;
  }

  static AnalysisKey Key;
};

AnalysisKey ControlDepGraph::Key;

// Helper function to perform DFS and compute the path condition for TargetBB
std::string getPathConditionHelper(BasicBlock *TargetBB,
                                   ControlDepGraph::Result &CDG,
                                   std::map<BasicBlock *, std::string> &Memo,
                                   std::set<BasicBlock *> &Visited) {
  // Base case: If the BasicBlock has no controllers, return Tautology
  if (CDG.find(TargetBB) == CDG.end() || CDG[TargetBB].empty()) {
    return "True";
  }

  // Check if the path condition for this block is already computed
  // (memoization)
  if (Memo.find(TargetBB) != Memo.end()) {
    return Memo[TargetBB];
  }

  // Check for cycles
  if (Visited.find(TargetBB) != Visited.end()) {
    return "True";
  }
  // Mark the current BasicBlock as visited
  Visited.insert(TargetBB);

  std::string PathCondition = "";

  // Iterate over all controlling nodes of TargetBB
  for (const auto &[ControllerBB, SuccIdx] : CDG[TargetBB]) {
    // Recursively find the path condition of the controlling BasicBlock
    std::string ControllerPathCond =
        getPathConditionHelper(ControllerBB, CDG, Memo, Visited);

    // Get the condition under which control transfers from ControllerBB to
    // TargetBB
    std::string TransferCond = getControlTransferCond(ControllerBB, SuccIdx);

    // Construct the condition as (path_cond(controllerBB) /\ TransferCond)
    std::string CombinedCond;
    if (ControllerPathCond == "True") {
      CombinedCond = TransferCond;
    } else {
      CombinedCond = "(" + ControllerPathCond + " /\\ " + TransferCond + ")";
    }

    // Accumulate using OR logic: PathCondition = PathCondition \/ CombinedCond
    if (!PathCondition.empty()) {
      PathCondition = PathCondition + " \\/ " + CombinedCond;
    } else {
      PathCondition = CombinedCond;
    }
  }

  // Store the computed result for this BasicBlock in the memoization map
  Memo[TargetBB] = PathCondition;
  // Remove the current BasicBlock from visited set as we backtrack
  Visited.erase(TargetBB);
  return PathCondition;
}

// Wrapper function to initiate the path condition finding process
std::string getPathCondition(BasicBlock *TargetBB,
                             ControlDepGraph::Result &CDG) {
  // Memoization map to avoid redundant calculations
  std::map<BasicBlock *, std::string> Memo;
  std::set<BasicBlock *> Visited;
  return getPathConditionHelper(TargetBB, CDG, Memo, Visited);
}

struct ControlDepGraphPrinter : public PassInfoMixin<ControlDepGraphPrinter> {
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
    RespJson["debug"]["request"] = ReqJson;

    auto *FPtr = findFunction(M, TargetFileName, TargetLineNumber);
    if (!FPtr) {
      OS << "Error: Could not find function on " << TargetFileName << ":"
         << TargetLineNumber << "\n";
      RespJson["error"]["message"] = ErrMsg;
      writeResponse(RespJson);
      return PreservedAnalyses::all();
    }
    auto &F = *FPtr;
    BasicBlock *TargetBB = findTargetBasicBlock(F, TargetLineNumber);
    if (!TargetBB) {
      OS << "Error: Could not find target basic block\n";
      RespJson["error"]["message"] = ErrMsg;
      writeResponse(RespJson);
      return PreservedAnalyses::all();
    }

    FunctionAnalysisManager &FAM =
        MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
    auto &CDG = FAM.getResult<ControlDepGraph>(F);

    RespJson["result"] = getPathCondition(TargetBB, CDG);
    writeResponse(RespJson);
    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

} // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "ControlDepGraph", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerAnalysisRegistrationCallback(
                [](FunctionAnalysisManager &FAM) {
                  FAM.registerPass([&] { return ControlDepGraph(); });
                });
            PB.registerPipelineParsingCallback(
                [](StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "print<control-dep-graph>") {
                    MPM.addPass(ControlDepGraphPrinter());
                    return true;
                  }
                  return false;
                });
          }};
}