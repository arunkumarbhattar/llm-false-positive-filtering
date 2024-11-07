import json
from enum import Enum
import os
import subprocess

def location_to_string(l) -> str:
    # 3.28 location object
    res = ""
    if 'id' in l:
        res += f"({l['id']}) "
    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', "Unknown")
    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', 0)
    startCol = l.get('physicalLocation', {}).get('region', {}).get('startColumn', 0)
    res += f"{uri}:{startLine}:{startCol}"
    msg = l.get('message', {}).get('text', None)
    if msg:
        res += ' ' + msg
    return res

class GroundTruth(Enum):
    GOOD = 1
    BAD = 2
    UNKNOWN = 3

class RepositoryManager:
    def __init__(self, repo_path, sarif_path, bitcode_path):
        self.repo_path = repo_path
        self.sarif_path = sarif_path
        self.bitcode_path = bitcode_path
        self.ctags_path = os.path.join(repo_path, "tags.txt")
        self.ctags = []
        self.ctags_by_filename = {}
        self.file_line_counts = {}
        self._load_ctags()
        self.compute_file_line_counts()

    def _gen_ctags(self):
        if os.path.exists(self.ctags_path):
            return

        print("Generating ctags...")
        command = [
            "ctags",
            "--languages=-all,+c,+c++",
            "--fields=+ne",
            "--kinds-c=f",
            "--kinds-c++=f",
            "-R",
            "."
        ]
        with open(self.ctags_path, 'w') as outfile:
            subprocess.run(command, cwd=self.repo_path, stdout=outfile, shell=False, check=True)

    def _load_ctags(self):
        self._gen_ctags()

        with open(self.ctags_path, 'r') as f:
            for line in f:
                if line.startswith('!'):
                    continue
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    continue  # Skip malformed lines
                func_name = parts[0]
                file_path = os.path.normpath(parts[1])
                kind_info = parts[3]

                # We're interested in functions ('f')
                if kind_info.startswith('f'):
                    start_line = None
                    end_line = None
                    for part in parts[3:]:
                        if part.startswith('line:'):
                            start_line = int(part.split(':')[1])
                        if part.startswith('end:'):
                            end_line = int(part.split(':')[1])
                    if start_line is not None:
                        if file_path not in self.ctags_by_filename:
                            self.ctags_by_filename[file_path] = []
                        self.ctags_by_filename[file_path].append((func_name, start_line, end_line))
                        self.ctags.append((func_name, file_path, start_line, end_line))

    def compute_file_line_counts(self):
        """
        Computes the total number of lines for each source file.
        """
        for src_filename in self.ctags_by_filename.keys():
            file_path = os.path.join(self.repo_path, src_filename)
            try:
                with open(file_path, 'r') as f:
                    self.file_line_counts[src_filename] = sum(1 for _ in f)
            except FileNotFoundError:
                print(f"File not found: {file_path}")
                self.file_line_counts[src_filename] = 0

    def get_codeql_results_and_gt(self, rule="cpp/uninitialized-local"):
        """
        Processes the SARIF file and retrieves CodeQL results along with ground truth data.
        """
        codeql_results = []
        try:
            with open(self.sarif_path, 'r') as f:
                sarif_data = json.load(f)
        except FileNotFoundError:
            print(f"SARIF file not found at {self.sarif_path}.")
            return codeql_results
        except json.JSONDecodeError as e:
            print(f"Error decoding SARIF file: {e}")
            return codeql_results

        for run in sarif_data.get('runs', []):
            for result in run.get('results', []):
                rule_id = result.get('ruleId', '')
                if rule_id != rule:
                    continue
                locations = result.get('locations', [])
                if not locations:
                    continue
                physical_location = locations[0].get('physicalLocation', {})
                artifact_location = physical_location.get('artifactLocation', {})
                src_filename = os.path.normpath(artifact_location.get('uri', ''))
                region = physical_location.get('region', {})
                startLine = region.get('startLine', 0)
                # Assuming 'msg' is part of the result message
                msg = result.get('message', {}).get('text', '')
                # Placeholder for ground truth (gt) retrieval logic
                gt = GroundTruth.UNKNOWN  # Replace with actual logic
                func_info = self.get_function_loc(src_filename, startLine)
                if func_info is None:
                    print(f"Skipping analysis for file: {src_filename}, line: {startLine} due to missing function location.")
                    continue
                func_name, func_startline, func_endline = func_info
                # Retrieve function code snippet
                func_code = self.get_function_code(src_filename, func_startline, func_endline)
                # 'rule_desc' should be retrieved based on 'rule_id'
                rule_desc = 'Description of the rule'  # Placeholder, replace with actual descriptions
                codeql_results.append((src_filename, startLine, msg, func_code, gt, rule_id, rule_desc))
        return codeql_results

    def get_function_code(self, filename, start_line, end_line):
        return self.dump_src(filename, start_line, end_line, print_lineno=True)

    def dump_src(self, filename, start, end, print_lineno):
        """
        Given a filename and start and end line numbers, return a string containing
        the file content from the start line to the end line (inclusive), with each
        line prepended by its line number.

        :param filename: The name of the file to read from.
        :param start: The starting line number (1-based).
        :param end: The ending line number (inclusive, 1-based).
        :param print_lineno: Whether to prepend line numbers.
        :return: A string containing the file content from start line to end line
                 with line numbers prepended.
        """
        file_path = os.path.join(self.repo_path, filename)
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return ""

        # Ensure the line numbers are within the valid range
        start = max(1, start)  # Ensure start is at least 1
        end = min(len(lines), end)  # Ensure end does not exceed the number of lines

        # Slice the list to get lines from start to end (inclusive)
        selected_lines = lines[start-1:end]

        if print_lineno:
            # Prepend line numbers
            numbered_lines = [f"{i+start}: {line}" for i, line in enumerate(selected_lines)]
            # Join the selected lines into a single string
            return ''.join(numbered_lines)
        else:
            return ''.join(selected_lines)

    def get_function_loc(self, src_filename, lineno):
        """
        Returns the function name, start line, and end line for a given file and line number.
        """
        src_filename = os.path.normpath(src_filename)
        relevant_ctags = self.ctags_by_filename.get(src_filename, [])
        if not relevant_ctags:
            print(f"No ctags data for file: {src_filename}")
            print(f"Available ctags filenames: {list(self.ctags_by_filename.keys())}")
            return None

        # Sort functions by start line
        sorted_funcs = sorted(relevant_ctags, key=lambda x: x[1])

        # Iterate through sorted functions to find the one that encompasses the given line
        for i, (func_name, start_line, end_line) in enumerate(sorted_funcs):
            # Determine end line if not provided
            if end_line is None:
                if i + 1 < len(sorted_funcs):
                    end_line = sorted_funcs[i + 1][1] - 1
                else:
                    end_line = self.file_line_counts.get(src_filename, lineno)  # Assume end at last line

            if start_line <= lineno <= end_line:
                return func_name, start_line, end_line

        # If no matching function is found
        print(f"No function found encompassing file: {src_filename}, line: {lineno}")
        return None

    # Additional methods like 'handle_llvm_tool_call' can remain unchanged
    # ...

if __name__ == '__main__':
    # Example usage
    repo_path = "test-proj"
    sarif_path = "cpp-codeql-results.sarif"
    bitcode_path = "test-proj/main.bc"

    repo = RepositoryManager(repo_path, sarif_path, bitcode_path)
    # Additional code for testing can be added here
