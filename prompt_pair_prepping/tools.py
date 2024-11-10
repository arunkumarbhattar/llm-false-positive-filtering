from pydantic import BaseModel, Field
from langchain.tools import tool
from prompts import repo
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

###################### Additional Tools #############################

# Tool 1: Get Function Arguments

class GetFunctionArgumentsInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file containing the function call.")
    lineno: int = Field(..., description="The line number in the source file where the function call occurs.")
    func_name: str = Field(..., description="The name of the function being called.")

@tool(args_schema=GetFunctionArgumentsInput)
def get_function_arguments(filename: str, lineno: int, func_name: str) -> str:
    """
    Retrieves the arguments passed to a function call at a specific location in the code.

    Use this tool when you need to find out what arguments are being passed to a function at a particular call site.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file containing the function call.
    lineno : int
        The line number in the source file where the function call occurs.
    func_name : str
        The name of the function being called.

    Returns:
    --------
    str
        A string containing the arguments passed to the function call.
    """
    return repo.handle_llvm_tool_call(get_function_arguments.name, filename=filename, lineno=lineno)


# Tool 2: Get Path Constraints

class GetPathConstraintsInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file to analyze.")
    lineno: int = Field(..., description="The line number in the source file for which the path constraints are to be determined.")

@tool(args_schema=GetPathConstraintsInput)
def get_path_constraints(filename: str, lineno: int) -> str:
    """
    Retrieves the path constraints (conditions) that must be true to reach a specific line in the code.

    Use this tool when you need to understand the conditions under which a particular line of code is executed.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file to analyze.
    lineno : int
        The line number in the source file where the path constraints are to be determined.

    Returns:
    --------
    str
        A string representation of the path constraints leading to the specified line.
    """
    return repo.handle_llvm_tool_call(get_path_constraints.name, filename=filename, lineno=lineno)

# Tool 3: Get Buffer Size

class GetBufferSizeInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file to analyze.")
    lineno: int = Field(..., description="The line number in the source file where the buffer variable is used.")
    varname: str = Field(..., description="The name of the buffer variable.")

@tool(args_schema=GetBufferSizeInput)
def get_buffer_size(filename: str, lineno: int, varname: str) -> str:
    """
    Determines the size of a buffer variable at a specific point in the code.

    Use this tool when you need to find out the size of a buffer to check for potential overflows.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file to analyze.
    lineno : int
        The line number in the source file where the buffer variable is used.
    varname : str
        The name of the buffer variable.

    Returns:
    --------
    str
        A string containing the size of the buffer.
    """
    return repo.handle_llvm_tool_call(get_buffer_size.name, filename=filename, lineno=lineno)

# Tool 4: Get Data Size

class GetDataSizeInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file to analyze.")
    lineno: int = Field(..., description="The line number in the source file where the data operation occurs.")
    varname: str = Field(..., description="The name of the data variable or buffer.")

@tool(args_schema=GetDataSizeInput)
def get_data_size(filename: str, lineno: int, varname: str) -> str:
    """
    Determines the size of the data being written to or read from a buffer variable at a specific point in the code.

    Use this tool when you need to understand the amount of data that may be stored in or retrieved from a buffer.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file to analyze.
    lineno : int
        The line number in the source file where the data operation occurs.
    varname : str
        The name of the data variable or buffer.

    Returns:
    --------
    str
        A string containing the data size or constraints on the data size.
    """
    return repo.handle_llvm_tool_call(get_data_size.name, filename=filename, lineno=lineno)

# Tool 5: Get Variable Usage Paths

class GetVariableUsagePathsInput(BaseModel):
    filename: str = Field(..., description="The name (and path, if necessary) of the source file containing the variable.")
    varname: str = Field(..., description="The name of the variable to analyze.")

@tool(args_schema=GetVariableUsagePathsInput)
def get_variable_usage_paths(filename: str, varname: str) -> str:
    """
    Identifies all paths where a variable is used after being defined, including any function calls on the variable.

    Use this tool when you need to check for potential issues like use-after-free or double-free.

    Parameters:
    -----------
    filename : str
        The name (and path, if necessary) of the source file containing the variable.
    varname : str
        The name of the variable to analyze.

    Returns:
    --------
    str
        A string describing the usage paths of the variable, including line numbers where it is used or freed.
    """
    return repo.handle_llvm_tool_call(get_variable_usage_paths.name, filename=filename)


###################### Tools metadata #############################
tool_list = [
    get_func_definition,
    variable_def_finder,
    get_path_constraint,
    get_function_arguments,
    get_path_constraints,
    get_buffer_size,
    get_data_size,
    get_variable_usage_paths
]

toolstr2callable = {tool.name: tool for tool in tool_list}
