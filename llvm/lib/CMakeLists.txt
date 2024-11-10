# THE LIST OF PLUGINS AND THE CORRESPONDING SOURCE FILES
# ======================================================
set(LLVM_TOOLS_PLUGINS
    ControlDepGraph
    VarDefFinder
    )

set(ControlDepGraph_SOURCES
  ControlDepGraph.cpp helper.cpp GetIfCond.cpp)
set(VarDefFinder_SOURCES
  VariableDefinitionFinderPass.cpp helper.cpp)

# CONFIGURE THE PLUGIN LIBRARIES
# ==============================
foreach( plugin ${LLVM_TOOLS_PLUGINS} )
    # Create a library corresponding to 'plugin'
    add_library(
      ${plugin}
      SHARED
      ${${plugin}_SOURCES}
      )

    # Configure include directories for 'plugin'
    target_include_directories(
      ${plugin}
      PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/../include"
    )

    target_link_libraries(
      ${plugin}
      "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
      )
endforeach()

target_link_libraries(ControlDepGraph clang)
