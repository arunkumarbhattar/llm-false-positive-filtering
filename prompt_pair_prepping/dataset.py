import json
from enum import Enum
import os
from collections import defaultdict
import subprocess

class GroundTruth(Enum):
    GOOD = 1
    BAD = 2
    UNRELATED = 3
    ERR = 4


class Dataset:
    def __init__(self, repo_path, sarif_path, bitcode_path):
        self.repo_path = repo_path
        self.sarif_path = sarif_path

        self.ctags_path = os.path.join(self.repo_path, "tags")
        self.ctags = []
        # self.ctags_by_filename is a dictionary where the key is the file_path (filename),
        # and the value is a list of tuples (identifier, start_line, end_line)
        self.ctags_by_filename = defaultdict(list)
        self._load_ctags()

        self.bitcode_path = bitcode_path

    def get_codeql_results_and_gt(self, rule="cpp/uninitialized-local"):
        with open(self.sarif_path, 'r') as f:
            s = json.load(f)

        codeql_results = []
        for run in s.get('runs', []):
            rules_metadata = run['tool']['driver']['rules']
            for r in run.get('results', []):
                # TODO: The ruleId field is optional and potentially ambiguous. We might have to fetch the actual
                # ruleId from the rule metadata via the ruleIndex field.
                # (see https://github.com/microsoft/sarif-tutorials/blob/main/docs/2-Basics.md#rule-metadata)
                ruleId = r['ruleId']
                ruleIndex = r['ruleIndex']
                #if uri is None or match_path_and_rule(uri, ruleId, args.patterns):
                rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                if rule_level not in ['error', 'warning']:
                    continue
                if rule is not None and ruleId != rule:
                    continue
                gt = r['groundTruth']
                if gt != 'TP' and gt != 'FP':
                    continue
                msg = r['message']['text']

                for l in r.get('locations', []):
                    # TODO: The uri field is optional. We might have to fetch the actual uri from "artifacts" via "index"
                    # (see https://github.com/microsoft/sarif-tutorials/blob/main/docs/2-Basics.md#-linking-results-to-artifacts)
                    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                    func_name, func_startline, func_endline = self.get_function_loc(uri, startLine)
                    func = self.dump_src(uri, func_startline, func_endline, True)
                    mygt = GroundTruth.BAD if gt == 'TP' else GroundTruth.GOOD
                    codeql_results.append((uri, startLine, msg, func, mygt))
        return codeql_results

    def get_codeql_results_and_gt_all_facts(self):
        with open(self.sarif_path, 'r') as f:
            s = json.load(f)

        codeql_results = []
        for run in s.get('runs', []):
            rules_metadata = run['tool']['driver']['rules']
            for r in run.get('results', []):
                # TODO: The ruleId field is optional and potentially ambiguous. We might have to fetch the actual
                # ruleId from the rule metadata via the ruleIndex field.
                # (see https://github.com/microsoft/sarif-tutorials/blob/main/docs/2-Basics.md#rule-metadata)
                ruleId = r['ruleId']
                ruleIndex = r['ruleIndex']
                #if uri is None or match_path_and_rule(uri, ruleId, args.patterns):
                rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                if rule_level not in ['error', 'warning']:
                    continue
                gt = r['groundTruth']
                if gt != 'TP' and gt != 'FP':
                    continue
                msg = r['message']['text']

                for l in r.get('locations', []):
                    # TODO: The uri field is optional. We might have to fetch the actual uri from "artifacts" via "index"
                    # (see https://github.com/microsoft/sarif-tutorials/blob/main/docs/2-Basics.md#-linking-results-to-artifacts)
                    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                    func_name, func_startline, func_endline = self.get_function_loc(uri, startLine)
                    func = self.dump_src(uri, func_startline, func_endline, True)
                    mygt = GroundTruth.BAD if gt == 'TP' else GroundTruth.GOOD
                    codeql_results.append((uri, startLine, msg, func, mygt, ruleId))
        return codeql_results

    def manually_label_codeql_results(self, new_sarif_path, rule="cpp/uninitialized-local"):
        with open(self.sarif_path, 'r') as f:
            s = json.load(f)

        for run in s.get('runs', []):
            rules_metadata = run['tool']['driver']['rules']
            for r in run.get('results', []):
                if 'groundTruth' in r:
                    continue
                ruleId = r['ruleId']
                ruleIndex = r['ruleIndex']
                rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                if rule_level not in ['error', 'warning']:
                    continue
                #if rule is not None and ruleId != rule:
                #    continue
                msg = r['message']['text']

                for l in r.get('locations', []):
                    uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                    startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                    startColumn = l.get('physicalLocation', {}).get('region', {}).get('startColumn', None)
                    print(f"\033[32m{ruleId}\033[0m")
                    print(msg)
                    print(f"{uri}:{startLine}:{startColumn}")
                    gt = input("Create ground truth label (t for TP, f for FP, n for N/A): ")
                    r['groundTruth'] = 'TP' if gt == 't' else ('FP' if gt == 'f' else 'N/A')
                    print()

        print("Writing new SARIF...")
        with open(new_sarif_path, 'w') as f:
            json.dump(s, f, indent=2)

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
                parts = line.split('\t')
                identifier = parts[0]
                file_path = parts[1]
                # Extract line numbers from the tags file
                start_line = None
                end_line = None
                for part in parts[2:]:
                    if part.startswith("line:"):
                        start_line = int(part.split(':')[1])
                    if part.startswith("end:"):
                        end_line = int(part.split(':')[1])
                self.ctags.append((identifier, file_path, start_line, end_line))
                self.ctags_by_filename[file_path].append((identifier, start_line, end_line))

    def get_function_definition(self, function_name):
        #dumped_code = ""
        for identifier, file_path, start_line, end_line in self.ctags:
            if identifier == function_name:
                return self.dump_src(file_path, start_line, end_line, False)

    def dump_src(self, filename, start, end, print_lineno):
        """
        Given a filename and start and end line numbers, return a string containing
        the file content from the start line to the end line (inclusive), with each
        line prepended by its line number.

        :param filename: The name of the file to read from.
        :param start: The starting line number (1-based).
        :param end: The ending line number (inclusive, 1-based).
        :return: A string containing the file content from start line to end line
                 with line numbers prepended.
        """
        with open(os.path.join(self.repo_path, filename), 'r') as file:
            lines = file.readlines()

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
        relevant_ctags = self.ctags_by_filename.get(src_filename, [])
        matched_funcs = [
            (identifier, start_line, end_line)
            for identifier, start_line, end_line in relevant_ctags
            if start_line <= lineno <= end_line
        ]
        if len(matched_funcs) == 1:
            return matched_funcs[0]
        else:
            print("Multiple or no match:", matched_funcs)

    def variable_def_finder(self, filename, lineno, varname):
        llvm_dir = os.environ.get('LLVM_DIR')
        if llvm_dir is None:
            return "Error: LLVM_DIR environment variable is not set."

        # Path to the 'opt' tool
        opt_path = os.path.join(llvm_dir, 'bin', 'opt')

        cmd = [
            opt_path,
            "-load-pass-plugin", "/home/arun/Desktop/llm-false-positive-filtering/llvm/build/lib/libVarDefFinder.so",
            "-passes=variable-def-finder",
            "-disable-output",
            self.bitcode_path
        ]

        # Create JSON request
        request_json = json.dumps({
            "args": {
                "filename": filename,
                "lineno": lineno,
                "variable-name": varname
            }
        })

        try:
            # Pass the JSON request via stdin
            result = subprocess.run(
                cmd,
                input=request_json,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("\n\nResult from running variable_def_finder pass is ---> \n")
            print(result.stdout)
            stderr_output = result.stderr
            return stderr_output
        except subprocess.CalledProcessError as e:
            # Capture stderr in case of an error
            return f"Command failed with error: {e.stderr}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"
    def get_path_constraint(self, filename, lineno):
        llvm_dir = os.environ.get('LLVM_DIR')
        if llvm_dir is None:
            return "Error: LLVM_DIR environment variable is not set."

        # Path to the 'opt' tool
        opt_path = os.path.join(llvm_dir, 'bin', 'opt')

        cmd = [
            opt_path,
            "-load-pass-plugin", "/home/arun/Desktop/llm-false-positive-filtering/llvm/build/lib/libControlDepGraph.so",
            "-passes=print<control-dep-graph>",
            "-disable-output",
            self.bitcode_path
        ]

        # Create JSON request
        request_json = json.dumps({
            "args": {
                "filename": filename,
                "lineno": lineno
            }
        })

        try:
            # Pass the JSON request via stdin
            result = subprocess.run(
                cmd,
                input=request_json,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            print("\n\nResult from running path constraint pass is ---> \n")
            print(result.stdout)

            stderr_output = result.stderr
            return stderr_output
        except subprocess.CalledProcessError as e:
            # Capture stderr in case of an error
            return f"Command failed with error: {e.stderr}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def get_variable_usage_paths(self, filename, varname):
        pass

    def get_data_size(self, filename, lineno, varname):
        pass

    def get_buffer_size(self, filename, lineno, varname):
        pass

    def get_path_constraints(self, filename, lineno):
        pass

    def get_function_arguments(self, filename, lineno, func_name):
        pass
