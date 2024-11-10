#include <clang-c/Index.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Struct to hold data for matching line, column, and resulting expression text
struct ClientData {
  unsigned lineNumber;
  unsigned columnNumber;
  std::string expressionText;
};

// Helper function to check if a line and column are within a specific
// CXSourceRange
bool isLocationWithinRange(CXSourceRange range, unsigned line,
                           unsigned column) {
  unsigned startLine, startColumn, endLine, endColumn;
  CXSourceLocation startLoc = clang_getRangeStart(range);
  CXSourceLocation endLoc = clang_getRangeEnd(range);

  clang_getFileLocation(startLoc, nullptr, &startLine, &startColumn, nullptr);
  clang_getFileLocation(endLoc, nullptr, &endLine, &endColumn, nullptr);

  return (line > startLine || (line == startLine && column >= startColumn)) &&
         (line < endLine || (line == endLine && column <= endColumn));
}

// Function to read text from the file based on a source range
std::string readRangeFromFile(CXSourceRange range) {
  CXSourceLocation startLoc = clang_getRangeStart(range);
  CXSourceLocation endLoc = clang_getRangeEnd(range);
  CXFile file;
  unsigned startOffset, endOffset;

  // Get the file location for the start and end of the range
  clang_getFileLocation(startLoc, &file, nullptr, nullptr, &startOffset);
  clang_getFileLocation(endLoc, &file, nullptr, nullptr, &endOffset);

  // std::ifstream inFile(filename);
  CXString filename = clang_getFileName(file);
  std::ifstream inFile(clang_getCString(filename));
  if (!inFile) {
    std::cerr << "Failed to open file: " << clang_getCString(filename)
              << std::endl;
    clang_disposeString(filename);
    return "";
  }
  clang_disposeString(filename);

  // Read the text from the specified range
  inFile.seekg(startOffset);
  std::vector<char> buffer(endOffset - startOffset);
  inFile.read(buffer.data(), buffer.size());

  return std::string(buffer.begin(), buffer.end());
}

// Check if a cursor is a binary logical operator (&& or ||)
bool isBinLogicOperator(CXCursor cursor) {
  if (clang_getCursorKind(cursor) == CXCursor_BinaryOperator) {
    // Get the binary operator kind
    CXBinaryOperatorKind opKind = clang_getCursorBinaryOperatorKind(cursor);

    // Check if it is logical AND (&&) or logical OR (||)
    return opKind == CXBinaryOperator_LAnd || opKind == CXBinaryOperator_LOr;
  }
  return false;
}

// Helper function to find the maximal subexpression without binary logical
// operations
void findMaximalSubexprWithoutBinLogicOps(CXCursor cursor, ClientData &data) {
  CXSourceRange range = clang_getCursorExtent(cursor);

  // Check if the cursor itself matches the location and is not a binary logical
  // operator
  if (isLocationWithinRange(range, data.lineNumber, data.columnNumber) &&
      !isBinLogicOperator(cursor)) {
    data.expressionText = readRangeFromFile(range);
    return;
  }

  // Recurse into children if the cursor itself is a binary logical operator
  clang_visitChildren(
      cursor,
      [](CXCursor child, CXCursor parent, CXClientData client_data) {
        auto &data = *static_cast<ClientData *>(client_data);
        CXSourceRange childRange = clang_getCursorExtent(child);

        if (isLocationWithinRange(childRange, data.lineNumber,
                                  data.columnNumber)) {
          if (!isBinLogicOperator(child)) {
            // Found matching expression without binary logical operators
            data.expressionText = readRangeFromFile(childRange);
            return CXChildVisit_Break;
          }
          // Continue recursing to find the maximal subexpression
          return CXChildVisit_Recurse;
        }
        return CXChildVisit_Continue;
      },
      &data);
}

// Main function to get the condition of if/while/do/for at line and column
std::string getControlCondAtLoc(std::string filename, unsigned lineNumber,
                                unsigned columnNumber) {
  CXIndex index = clang_createIndex(0, 0);
  CXTranslationUnit translationUnit = clang_parseTranslationUnit(
      index, filename.c_str(), nullptr, 0, nullptr, 0, CXTranslationUnit_None);

  if (!translationUnit) {
    std::cerr << "Unable to parse translation unit!" << std::endl;
    clang_disposeIndex(index);
    return "";
  }

  ClientData data = {lineNumber, columnNumber, ""};
  CXCursor cursor = clang_getTranslationUnitCursor(translationUnit);

  clang_visitChildren(
      cursor,
      [](CXCursor c, CXCursor parent, CXClientData client_data) {
        auto &data = *static_cast<ClientData *>(client_data);

        // Look for IfStmt/WhileStmt/... with the specified line and column
        // number
        CXCursorKind parentKind = clang_getCursorKind(parent);
        if ((parentKind == CXCursor_IfStmt ||
             parentKind == CXCursor_WhileStmt ||
             parentKind == CXCursor_DoStmt || parentKind == CXCursor_ForStmt ||
             parentKind == CXCursor_SwitchStmt) &&
            clang_isExpression(clang_getCursorKind(c))) {
          // Search for maximal subexpression without binary logical operators
          findMaximalSubexprWithoutBinLogicOps(c, data);
          // return CXChildVisit_Break;
        }
        return CXChildVisit_Recurse;
      },
      &data);

  clang_disposeTranslationUnit(translationUnit);
  clang_disposeIndex(index);

  return data.expressionText;
}