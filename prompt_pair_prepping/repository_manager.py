import json
from enum import Enum
import os
from collections import defaultdict
import subprocess


def location_to_string(l) -> str:
    """
    Converts a SARIF location object into a human-readable string.

    Args:
        l (dict): A dictionary representing a location object in SARIF format.

    Returns:
        str: A formatted string containing the URI, line number, column number, and message (if any).
    """
    res = ""
    # Include location ID if present
    if 'id' in l:
        res += f"({l['id']}) "

    # Extract URI, start line, and start column from the location object
    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', "Unknown")
    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', 0)
    startCol = l.get('physicalLocation', {}).get('region', {}).get('startColumn', 0)

    # Append URI, line number, and column number to the result string
    res += f"{uri}:{startLine}:{startCol}"

    # Append message text if available
    msg = l.get('message', {}).get('text', None)
    if msg:
        res += ' ' + msg

    return res


def location_array_to_string(locations):
    """
    Converts an array of SARIF location objects into a formatted string.

    Args:
        locations (list): A list of location objects in SARIF format.

    Returns:
        str: A concatenated string of all formatted location strings, each prefixed with two spaces.
    """
    res = ""
    for l in locations:
        res += "  " + location_to_string(l) + '\n'
    return res


def codeFlows_to_string(codeFlows):
    """
    Converts SARIF codeFlows into a formatted string representing the data flow path.

    Args:
        codeFlows (list): A list of codeFlow objects in SARIF format.

    Returns:
        str: A concatenated string of all formatted code flow locations, each prefixed with two spaces.
             Returns an empty string if codeFlows is malformed or missing expected fields.
    """
    try:
        res = ""
        # Iterate over the first codeFlow's threadFlows and their locations
        for tfloc in codeFlows[0]['threadFlows'][0]['locations']:
            res += "  " + location_to_string(tfloc['location']) + '\n'
        return res
    except (TypeError, IndexError, KeyError):
        # Return empty string if codeFlows structure is unexpected
        return ""


class GroundTruth(Enum):
    """
    Enumeration representing the ground truth labels for CodeQL alerts.
    """
    GOOD = 1       # Indicates the alert is a false positive
    BAD = 2        # Indicates the alert is a true positive
    UNRELATED = 3  # Indicates the alert is unrelated
    ERR = 4        # Indicates an error in processing


class RepositoryManager:
    """
    Manages a CodeQL repository, processes SARIF files, and provides utilities to interact with code artifacts.

    Attributes:
        repo_path (str): The base path to the Juliet test suite repository.
        sarif_path (str): The path to the SARIF file containing CodeQL results.
        bitcode_path (str): The path to the bitcode file associated with the SARIF results.
        ctags_path (str): The path to the ctags file within the repository.
        ctags (list): A list of tuples containing ctags information (identifier, file path, start line, end line).
        ctags_by_filename (defaultdict): A dictionary mapping filenames to lists of ctags tuples.
    """

    def __init__(self, repo_path, sarif_path, bitcode_path):
        """
        Initializes the RepositoryManager with repository paths and loads ctags data.

        Args:
            repo_path (str): The base path to the Juliet test suite repository.
            sarif_path (str): The path to the SARIF file containing CodeQL results.
            bitcode_path (str): The path to the bitcode file associated with the SARIF results.
        """
        self.repo_path = repo_path
        self.sarif_path = sarif_path

        self.ctags_path = os.path.join(self.repo_path, "tags")
        self.ctags = []
        # Initializes a dictionary where each key is a filename and the value is a list of ctags tuples
        self.ctags_by_filename = defaultdict(list)
        self._load_ctags()

        self.bitcode_path = bitcode_path

    def get_codeql_results_and_gt(self):
        """
        Parses the SARIF file to extract CodeQL results along with their ground truth labels.

        Returns:
            list: A list of tuples containing alert details:
                  (source file URI, line number, message, function code, ground truth label, rule ID, rule description)
        """
        with open(self.sarif_path, 'r') as f:
            s = json.load(f)

        codeql_results = []
        for run in s.get('runs', []):
            rules_metadata = run['tool']['driver']['rules']
            for r in run.get('results', []):
                ruleId = r['ruleId']
                ruleIndex = r['ruleIndex']
                # Extract rule level and description from rules metadata
                rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                rule_desc = rules_metadata[ruleIndex]['fullDescription']['text']
                # Consider only rules with level 'error' or 'warning'
                if rule_level not in ['error', 'warning']:
                    continue
                # Extract ground truth label from properties
                gt = r.get('properties', {}).get('groundTruth', 'N/A')
                if gt != 'TP' and gt != 'FP':
                    continue
                # Extract message text
                msg = r['message']['text']
                # Append related locations if present
                if 'relatedLocations' in r:
                    msg += '\n' + location_array_to_string(r['relatedLocations'])
                # Append code flows if present
                if 'codeFlows' in r:
                    msg += "Dataflow path related to this alert:\n"
                    msg += codeFlows_to_string(r['codeFlows'])

                for l in r.get('locations', []):
                    # Extract URI and line number from the location object
                    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                    # Retrieve function details based on URI and line number
                    func_name, func_startline, func_endline = self.get_function_loc(uri, startLine)
                    func = self.dump_src(uri, func_startline, func_endline, True)
                    # Map ground truth to enumeration
                    mygt = GroundTruth.BAD if gt == 'TP' else GroundTruth.GOOD
                    # Append the alert details to the results list
                    codeql_results.append((uri, startLine, msg, func, mygt, ruleId, rule_desc))
        return codeql_results

    def manually_label_codeql_results(self, new_sarif_path):
        """
        Provides an interactive way to manually label CodeQL results in the SARIF file.

        Args:
            new_sarif_path (str): The path where the newly labeled SARIF file will be saved.
        """
        with open(self.sarif_path, 'r') as f:
            s = json.load(f)

        for run in s.get('runs', []):
            rules_metadata = run['tool']['driver']['rules']
            for r in run.get('results', []):
                r['properties'] = r.get('properties', {})
                if 'groundTruth' in r['properties']:
                    # Skip if groundTruth is already labeled
                    continue
                ruleId = r['ruleId']
                ruleIndex = r['ruleIndex']
                rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                # Consider only rules with level 'error' or 'warning'
                if rule_level not in ['error', 'warning']:
                    continue
                msg = r['message']['text']

                gt = None
                for l in r.get('locations', []):
                    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                    startColumn = l.get('physicalLocation', {}).get('region', {}).get('startColumn', None)
                    print(f"\033[32m{ruleId}\033[0m")
                    print(msg)
                    print(f"{uri}:{startLine}:{startColumn}")
                    src_code = self.dump_src(uri, startLine, startLine, True)
                    print(src_code)
                    # Automatically mark CodeQL test cases based on 'BAD' or 'GOOD' comments in the source code
                    if 'BAD' in src_code:
                        gt = 'TP'
                    elif 'GOOD' in src_code:
                        gt = 'FP'
                if not gt:
                    # Prompt user for ground truth label if not automatically determined
                    x = input("Create ground truth label (t for TP, f for FP): ").lower()
                    gt = 'TP' if x == 't' else ('FP' if x == 'f' else 'N/A')
                print(gt)
                r['properties']['groundTruth'] = gt
                print()

        print("Writing new SARIF...")
        with open(new_sarif_path, 'w') as f:
            json.dump(s, f, indent=2)

    def _gen_ctags(self):
        """
        Generates a ctags file for the repository if it does not already exist.
        """
        if os.path.exists(self.ctags_path):
            # Ctags file already exists; no need to generate
            return

        print("Generating ctags...")
        command = [
            "ctags",
            "--languages=-all,+c,+c++",     # Include only C and C++ languages
            "--fields=+ne",                 # Include extra fields: name and extension
            "--kinds-c=f",                  # Include only function definitions for C
            "--kinds-c++=f",                # Include only function definitions for C++
            "-R",                           # Recursive directory traversal
            "."                             # Current directory
        ]
        with open(self.ctags_path, 'w') as outfile:
            # Execute the ctags command within the repository path and write output to ctags file
            subprocess.run(command, cwd=self.repo_path, stdout=outfile, shell=False, check=True)

    def _load_ctags(self):
        """
        Loads ctags data into memory by generating the ctags file if necessary and parsing its contents.
        """
        self._gen_ctags()

        with open(self.ctags_path, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    # Skip lines starting with '!', which are typically headers in ctags files
                    continue
                parts = line.split('\t')
                identifier = parts[0]
                file_path = parts[1]
                # Initialize line numbers
                start_line = None
                end_line = None
                # Extract line number information from ctags fields
                for part in parts[2:]:
                    if part.startswith("line:"):
                        start_line = int(part.split(':')[1])
                    if part.startswith("end:"):
                        end_line = int(part.split(':')[1])
                # Append ctags information to the list and dictionary
                self.ctags.append((identifier, file_path, start_line, end_line))
                self.ctags_by_filename[file_path].append((identifier, start_line, end_line))

    def get_function_definition(self, function_name):
        """
        Retrieves the source code definition of a specified function.

        Args:
            function_name (str): The name of the function whose definition is to be retrieved.

        Returns:
            str: The source code of the function, or an empty string if not found.
        """
        for identifier, file_path, start_line, end_line in self.ctags:
            if identifier == function_name:
                return self.dump_src(file_path, start_line, end_line, False)
        return ""

    def dump_src(self, filename, start, end, print_lineno):
        """
        Extracts and returns a portion of the source code from a specified file.

        Args:
            filename (str): The name of the file to read from.
            start (int): The starting line number (1-based).
            end (int): The ending line number (inclusive, 1-based).
            print_lineno (bool): If True, prepends line numbers to each line.

        Returns:
            str: A string containing the selected source code lines, optionally with line numbers.
        """
        with open(os.path.join(self.repo_path, filename), 'r') as file:
            lines = file.readlines()

        # Ensure the line numbers are within the valid range
        start = max(1, start)  # Ensure start is at least 1
        end = min(len(lines), end)  # Ensure end does not exceed the number of lines

        # Slice the list to get lines from start to end (inclusive)
        selected_lines = lines[start-1:end]

        if print_lineno:
            # Prepend line numbers to each line
            numbered_lines = [f"{i+start}: {line}" for i, line in enumerate(selected_lines)]
            return ''.join(numbered_lines)
        else:
            # Return lines without line numbers
            return ''.join(selected_lines)

    def get_function_loc(self, src_filename, lineno):
        """
        Finds the location (start and end lines) of the function containing a specific line in a source file.

        Args:
            src_filename (str): The name of the source file.
            lineno (int): The line number within the source file.

        Returns:
            tuple: A tuple containing (function name, start line, end line).
                   Returns (None, None, None) if no matching function is found.
        """
        relevant_ctags = self.ctags_by_filename.get(src_filename, [])
        matched_funcs = [
            (identifier, start_line, end_line)
            for identifier, start_line, end_line in relevant_ctags
            if start_line <= lineno <= end_line
        ]
        if len(matched_funcs) == 1:
            return matched_funcs[0]
        else:
            # Print a message if multiple or no matches are found
            print("Multiple or no match:", matched_funcs)
            return (None, None, None)

    def handle_llvm_tool_call(self, tool_name, **kwargs):
        """
        Handles calls to LLVM-based tools by executing the appropriate LLVM pass.

        Args:
            tool_name (str): The name of the tool to invoke (e.g., 'variable_def_finder').
            **kwargs: Additional keyword arguments required by the tool.

        Returns:
            str: The result produced by the tool, or an error message if execution fails.
        """
        request = {
            "method": tool_name,
            "args": kwargs
        }

        # Retrieve LLVM directories from environment variables
        llvm_dir = os.environ.get('LLVM_DIR')
        if llvm_dir is None:
            return "Error: LLVM_DIR environment variable is not set."
        shared_lib_dir = os.environ.get('LLVM_PASSES_LIB_DIR')
        if shared_lib_dir is None:
            return "Error: LLVM_PASSES_LIB_DIR environment variable is not set."

        # Define the path to the LLVM 'opt' executable
        opt_path = os.path.join(llvm_dir, 'bin', 'opt')  # $LLVM_DIR/bin/opt

        # Mapping of tool names to their corresponding LLVM shared libraries and pass names
        llvm_passes_info = {
            "variable_def_finder": ("libVarDefFinder.so", "variable-def-finder"),
            "get_path_constraint": ("libControlDepGraph.so", "print<control-dep-graph>"),
        }
        # Retrieve pass information based on the tool name
        pass_info = llvm_passes_info.get(tool_name)
        if pass_info is None:
            return f"Error: No LLVM pass configured for tool '{tool_name}'."

        # Construct the command to invoke the LLVM pass
        cmd = [
            opt_path,
            "-load-pass-plugin", os.path.join(shared_lib_dir, pass_info[0]),
            f"-passes={pass_info[1]}",
            "-disable-output",
            self.bitcode_path
        ]

        # Define paths for named pipes used for inter-process communication
        request_pipe = '/tmp/request_pipe'
        response_pipe = '/tmp/response_pipe'

        # Create the named pipes if they do not exist
        if not os.path.exists(request_pipe):
            os.mkfifo(request_pipe)
        if not os.path.exists(response_pipe):
            os.mkfifo(response_pipe)

        # Start the LLVM 'opt' process
        process = subprocess.Popen(
            cmd,
            # stderr=subprocess.PIPE  # Optionally capture stderr for debugging
        )

        # Write the request to the request pipe in JSON format
        with open(request_pipe, 'w') as pipe:
            json.dump(request, pipe)

        # Read the response from the response pipe
        with open(response_pipe, 'r') as pipe:
            response = json.load(pipe)
            # Parse the response to extract the result
            if 'result' in response:
                result = str(response['result'])
            else:
                # Handle error messages from the tool
                msg = response['error']['message']
                result = f"Error: {msg}"

        # Clean up by removing the named pipes
        os.remove(request_pipe)
        os.remove(response_pipe)

        return result
