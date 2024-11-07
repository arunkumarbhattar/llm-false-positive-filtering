#include "helper.h"
#include <fstream>

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"

using namespace llvm;

bool inTargetFile(const Function &F, StringRef TargetFileName) {
  if (auto *SP = F.getSubprogram()) {
    if (auto *File = SP->getFile()) {
      StringRef FileName = File->getFilename();
      if (FileName.endswith(TargetFileName) ||
          TargetFileName.endswith(FileName)) {
         errs() << "Found file " << FileName << "\n";
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
           errs() << "Found line " << LineNum << "\n";
          return &BB;
        }
      }
    }
  }
  return nullptr;
}

Function *findFunction(Module &M, StringRef FileName, unsigned LineNum) {
  for (auto &F : M) {
    if (auto *SP = F.getSubprogram()) {
      if (auto *File = SP->getFile()) {
        StringRef ThisFileName = File->getFilename();
        if (ThisFileName.endswith(FileName) ||
            FileName.endswith(ThisFileName)) {
          // errs() << "Found file " << FileName << "\n";
          if (SP->getLine() == LineNum) {
            return &F;
          }
          // Check if line number of an instruction matches LineNum
          if (findTargetBasicBlock(F, LineNum)) {
            return &F;
          }
        }
      }
    }
  }
  return nullptr;
}

bool readRequest(nlohmann::json &J) {
  // Open the named pipe (FIFO) for reading
  std::ifstream request_pipe("/tmp/request_pipe");
  // Check if the request_pipe was opened successfully
  if (!request_pipe.is_open()) {
    return false;
  }
  // Read JSON data from the named pipe
  request_pipe >> J;
  request_pipe.close();
  return true;
}

bool writeResponse(const nlohmann::json &J) {
  // std::error_code EC;
  // llvm::raw_fd_ostream response_pipe("/tmp/response_pipe", EC);
  // if (EC) {
  //   return false;
  // }
  std::ofstream pipe("/tmp/response_pipe");
  if (!pipe.is_open()) {
    return false;
  }
  pipe << J;
  pipe.close();
  return true;
}
