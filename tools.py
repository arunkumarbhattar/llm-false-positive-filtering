from pydantic import BaseModel, Field
from langchain.tools import tool

###################### LLVM tools #############################

class FindVarDefInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file to analyze.")
    lineno: int = Field(..., description="The line number in the source file where the function containing the variable is defined.")
    varname: str = Field(..., description="The name of the local variable whose definitions are to be found.")

@tool(args_schema=FindVarDefInput)
def variable_def_finder(filename: str, lineno: int, varname: str) -> str:
    """
    Finds all definitions of a local variable in a specified function within a given file.

    Returns:
    --------
    list[int]
        A list of the line numbers of all definitions of the specified
        variable. If no definitions are found, the function returns an empty list.
    """
    from llm_triage import repo
    return repo.handle_llvm_tool_call(variable_def_finder.name, filename=filename, lineno=lineno, varname=varname)

class GetPathConstraintInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file to analyze.")
    lineno: int = Field(..., description="The line number in the source file where the function containing the variable is defined.")

@tool(args_schema=GetPathConstraintInput)
def get_path_constraint(filename: str, lineno: int) -> str:
    """
    Retrieves the path constraint for a specific source code location.

    This function analyzes the code at the given file and line number, extracting the
    logical path constraint that leads to the specified location. Path constraints
    are the conditions that must be true for the execution to reach the given line
    in the code.

    Returns:
    --------
    str
        A string representation of the path constraint that leads to the specified line.
        If no constraint is found or the analysis fails, an empty string or a relevant
        error message may be returned.
    """
    from llm_triage import repo
    return repo.handle_llvm_tool_call(get_path_constraint.name, filename=filename, lineno=lineno)

###################### Python tools #############################

class GetFuncDefInput(BaseModel):
    func_name: str = Field(..., description="The name of the function whose definition is to be retrieved.")

@tool(args_schema=GetFuncDefInput)
def get_func_definition(func_name: str) -> str:
    """Get the definition of a function.
    Retrieves the definition of a function as a string based on the function's name.

    Returns:
       A string representation of the function's definition, including the function
       signature and body.
    """
    from llm_triage import repo
    res = repo.get_function_definition(func_name)
    if not res or len(res) == 0:
        return f"The definition of {func_name} is unavailable."
    return res[:10000]

###################### Tools metadata #############################
tool_list = [
    get_func_definition,
    variable_def_finder,
    get_path_constraint
]

toolstr2callable  = {tool.name: tool for tool in tool_list}

##################### Testing ################################
if __name__ == "__main__":
    for tool in tool_list:
        if tool.args_schema:
            print(f"Schema for tool '{tool.name}':")
            print(tool.args_schema.model_json_schema())
        else:
            print(f"Tool '{tool.name}' has no args schema.")
