cmake_minimum_required(VERSION 3.20)
project(llvm-tools)

#===============================================================================
# 1. VERIFY LLVM INSTALLATION DIR
# This is just a bit of a sanity checking.
#===============================================================================
set(LT_LLVM_INSTALL_DIR "" CACHE PATH "LLVM installation directory")

# 1.1 Check the "include" directory
set(LT_LLVM_INCLUDE_DIR "${LT_LLVM_INSTALL_DIR}/include/llvm")
if(NOT EXISTS "${LT_LLVM_INCLUDE_DIR}")
  message(FATAL_ERROR "LT_LLVM_INSTALL_DIR (${LT_LLVM_INCLUDE_DIR}) is invalid.")
endif()

# 1.2 Check that the LLVMConfig.cmake file exists (the location depends on the OS)
set(LT_VALID_INSTALLATION FALSE)

# Ubuntu + Darwin
if(EXISTS "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
  set(LT_VALID_INSTALLATION TRUE)
endif()

# Fedora
if(EXISTS "${LT_LLVM_INSTALL_DIR}/lib64/cmake/llvm/LLVMConfig.cmake")
  set(LT_VALID_INSTALLATION TRUE)
endif()

if(NOT LT_VALID_INSTALLATION)
  message(FATAL_ERROR "LLVM installation directory (${LT_LLVM_INSTALL_DIR}) is invalid. Couldn't find LLVMConfig.cmake.")
endif()

#===============================================================================
# 2. LOAD LLVM CONFIGURATION
#    For more: http://llvm.org/docs/CMake.html#embedding-llvm-in-your-project
#===============================================================================
# Add the location of LLVMConfig.cmake to CMake search paths (so that
# find_package can locate it)
list(APPEND CMAKE_PREFIX_PATH "${LT_LLVM_INSTALL_DIR}/lib/cmake/llvm/")

# The way LLVMConfigVersion.cmake is set up, it will only match MAJOR.MINOR
# exactly, even if we do not specify "REQUIRED" in the statement below.
# So we accept any version and do the proper ranged check below.
find_package(LLVM CONFIG)

# We defer the version checking to this statement
if("${LLVM_VERSION_MAJOR}" VERSION_LESS 17)
  message(FATAL_ERROR "Found LLVM ${LLVM_VERSION_MAJOR}, but need LLVM 17 or above")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LT_LLVM_INSTALL_DIR}")

message("LLVM STATUS:
  Definitions ${LLVM_DEFINITIONS}
  Includes    ${LLVM_INCLUDE_DIRS}
  Libraries   ${LLVM_LIBRARY_DIRS}
  Targets     ${LLVM_TARGETS_TO_BUILD}"
)

# Set the LLVM header and library paths
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

#===============================================================================
# 3. LLVM-TOOLS BUILD CONFIGURATION
#===============================================================================
# Use the same C++ standard as LLVM does
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# Build type
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type (default Debug):" FORCE)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fdiagnostics-color=always")

# LLVM is normally built without RTTI. Be consistent with that.
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# -fvisibility-inlines-hidden is set when building LLVM and on Darwin warnings
# are triggered if llvm-tools is built without this flag (though otherwise it
# builds fine). For consistency, add it here too.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fvisibility-inlines-hidden" SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG)
if (${SUPPORTS_FVISIBILITY_INLINES_HIDDEN_FLAG})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility-inlines-hidden")
endif()

# Additional Flags for Debugging
# Ensure that debug symbols are generated and optimizations are disabled
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

# Optionally, add more detailed debugging information
# set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -ggdb")

# Set the build directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

#===============================================================================
# 4. ADD SUB-TARGETS
# Doing this at the end so that all definitions and link/include paths are
# available for the sub-projects.
#===============================================================================
add_subdirectory(lib)
