#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include <map>
#include <set>

using namespace llvm;

static cl::opt<std::string> TargetFileName(
    "filename",
    cl::desc("Source file name"),
    cl::value_desc("string")
    // cl::init("a.c")
);
static cl::opt<unsigned> TargetLineNumber(
    "lineno",
    cl::desc("Line number in the source file"),
    cl::value_desc("unsigned integer")
    // cl::init(0)
);

namespace {
// Helper function to get the start and end line numbers of a basic block
std::pair<unsigned, unsigned> getLineRange(const BasicBlock *BB) {
    unsigned startLine = UINT_MAX;
    unsigned endLine = 0;

    if (!BB) return {0, 0};

    for (const Instruction &I : *BB) {
        if (const DebugLoc &DL = I.getDebugLoc()) {
            unsigned line = DL.getLine();
            startLine = std::min(startLine, line);
            endLine = std::max(endLine, line);
        }
    }

    if (startLine == UINT_MAX) {
        // No debug information found
        return {0, 0};
    }

    return {startLine, endLine};
}

unsigned getControlTransferCond(BasicBlock *BB, unsigned SuccIdx) {
  auto *Terminator = BB->getTerminator();
  unsigned LineNum = 0;
  if (auto *BI = dyn_cast<BranchInst>(Terminator)) {
    if (BI->isConditional()) {
      if (auto *CondInst = dyn_cast<Instruction>(BI->getCondition())) {
        if (const DebugLoc &DL = CondInst->getDebugLoc()) {
          LineNum = DL.getLine();
        }
      }
      // SuccIdx == 0 means BRANCH IS TAKEN
      // SuccIdx == 1 means BRANCH IS NOT TAKEN
      assert(SuccIdx == 0 || SuccIdx == 1);
      if (SuccIdx == 0) {
        errs() << "Branch on line " << LineNum << " is TAKEN\n";
      } else {
        errs() << "Branch on line " << LineNum << " is NOT TAKEN\n";
      }
    }
  } else if (auto *SI = dyn_cast<SwitchInst>(Terminator)) {
    if (auto *CondInst = dyn_cast<Instruction>(SI->getCondition())) {
      if (const DebugLoc &DL = CondInst->getDebugLoc()) {
        LineNum = DL.getLine();
      }
    }
    if (SuccIdx == 0) {
      errs() << "Switch statement on line " << LineNum << " takes default case\n";
    } else if (ConstantInt *CaseValue = SI->findCaseDest(SI->getSuccessor(SuccIdx))) {
      errs() << "Switch condition on line " << LineNum << " is equal to "
        << CaseValue->getValue() << "\n";
    }
  }
  return LineNum;
}

bool inTargetFile(const Function &F, StringRef TargetFileName) {
    if (auto *SP = F.getSubprogram()) {
      if (auto *File = SP->getFile()) {
        StringRef FileName = File->getFilename();
        //if (FileName == TargetFileName) {
        if (FileName.endswith(TargetFileName) || TargetFileName.endswith(FileName)) {
          return true;
        }
      }
    }
    return false;
}

BasicBlock *findTargetBasicBlock(Function &F, unsigned LineNum) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto DL = I.getDebugLoc()) {
          if (DL.getLine() == LineNum) {
            return &BB;
          }
        }
      }
    }
    return nullptr;
}

struct ControlDepGraph : public AnalysisInfoMixin<ControlDepGraph> {
  // Credit: https://github.com/ARISTODE/program-dependence-graph/blob/main/src/ControlDependencyGraph.cpp
  using Result = std::map<BasicBlock*, std::set<std::pair<BasicBlock*, unsigned>>>;
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
            auto *NearestCommonDominator = PDT.findNearestCommonDominator(&BB, Succ);
            if (NearestCommonDominator == &BB) {
              // BB controls Succ through BB's SuccIdx-th successor
              //CDG[&BB].insert(Succ);
              CDG[Succ].insert({&BB, SuccIdx});
            }
            for (auto *Cur = PDT.getNode(Succ); Cur != PDT.getNode(NearestCommonDominator); Cur = Cur->getIDom()) {
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

struct ControlDepGraphPrinter : public PassInfoMixin<ControlDepGraphPrinter> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    BasicBlock *TargetBB = nullptr;

    if (!inTargetFile(F, TargetFileName) || !(TargetBB = findTargetBasicBlock(F, TargetLineNumber))) {
      return PreservedAnalyses::all();
    }
    errs() << "Found function " << F.getName() << " on line " << TargetLineNumber << " in file " << TargetFileName << "\n";

    auto &CDG = FAM.getResult<ControlDepGraph>(F);

    // errs() << "Control Dependence Graph for function " << F.getName() << ":\n";
    // for (auto &Entry : CDG) {
    //   auto EntryRange = getLineRange(Entry.first);
    //   errs() << "  Lines " << EntryRange.first << "-" << EntryRange.second << " depend on:\n";
    //   for (auto &Controller : Entry.second) {
    //     unsigned ControllerLine = getBranchCondLine(Controller.first);
    //     errs() << "    Lines " << ControllerLine << " via " << Controller.second << "\n";
    //     //auto DependentRange = getLineRange(Dependent);
    //     //errs() << "    Lines " << DependentRange.first << "-" << DependentRange.second << "\n";
    //   }
    // }

    auto *CurBB = TargetBB;
    do {
      auto It = CDG.find(CurBB);
      BasicBlock *ControllerBB = nullptr;
      unsigned ControllerBBSuccIdx;
      if (It != CDG.end()) {
        auto &Controllers = It->second;
        if (!Controllers.empty()) {
          assert(Controllers.size() == 1);
          // Get the first element of Controller
          std::tie(ControllerBB, ControllerBBSuccIdx) = *Controllers.begin();
        }
      }
      if (ControllerBB) {
        // Print ControllerBB
        unsigned ControllerLine = getControlTransferCond(ControllerBB, ControllerBBSuccIdx);
        // errs() << "  Controlled by Line " << ControllerLine << " via its successor " << ControllerBBSuccIdx << "\n";
        (void) ControllerLine;
      }
      CurBB = ControllerBB;
    } while (CurBB);

    return PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};

}  // end anonymous namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "ControlDepGraph", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerAnalysisRegistrationCallback(
        [](FunctionAnalysisManager &FAM) {
          FAM.registerPass([&] { return ControlDepGraph(); });
        });
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "print<control-dep-graph>") {
            FPM.addPass(ControlDepGraphPrinter());
            return true;
          }
          return false;
        });
    }
  };
}
