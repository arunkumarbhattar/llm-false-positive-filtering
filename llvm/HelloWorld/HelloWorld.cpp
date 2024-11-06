#include "llvm/IR/Function.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

// Define a command-line option
static cl::opt<std::string> InputOption(
    "my-option", cl::desc("A string option for the pass"),
    cl::value_desc("string"),
    cl::init("default-value") // Default value if the option is not provided
);

namespace {

struct CmdLineArgPass : public PassInfoMixin<CmdLineArgPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    // Access the command-line argument
    std::string optionValue = InputOption;

    // Print the command-line argument value
    errs() << "Command-line option value: " << optionValue << "\n";

    // Example: Perform different actions based on the input
    if (optionValue == "special") {
      errs() << "Special action triggered!\n";
    }

    return PreservedAnalyses::all();
  }
};

} // end anonymous namespace

// Register the pass with the new pass manager
llvm::PassPluginLibraryInfo getCmdLineArgPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CmdLineArgPass", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "cmd-line-arg-pass") {
                    FPM.addPass(CmdLineArgPass());
                    return true;
                  }
                  return false;
                });
          }};
}

// Export the pass to be used with `opt`
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getCmdLineArgPassPluginInfo();
}
