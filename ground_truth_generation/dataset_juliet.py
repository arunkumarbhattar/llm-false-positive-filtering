import json
import re
from enum import Enum
import os
import argparse


def full_path(rel_path):
    """
    Constructs the absolute path by joining the base directory with a relative path.

    Args:
        rel_path (str): The relative path to be joined with the base directory.

    Returns:
        str: The absolute path resulting from the combination of the base directory and the relative path.
    """
    return os.path.join(BASE_DIR, rel_path)


def parse_codeql_sarif(filename, rule=None):
    """
    Parses a CodeQL SARIF file to extract relevant CodeQL results based on specified rules.

    Args:
        filename (str): The name of the SARIF file to be parsed.
        rule (str, optional): The specific CodeQL rule ID to filter results. If None, all relevant rules are included.

    Returns:
        list: A list of tuples, each containing:
              (URI of the source file, line number of the alert, alert message)
    """
    with open(full_path(filename), 'r') as f:
        s = json.load(f)

    codeql_results = []
    for run in s.get('runs', []):
        rules_metadata = run['tool']['driver']['rules']
        if run.get('results', []):
            for r in run['results']:
                if r.get('locations', []):
                    for l in r['locations']:
                        uri = l.get('physicalLocation', {}).get('artifactLocation', {}).get('uri', None)
                        startLine = l.get('physicalLocation', {}).get('region', {}).get('startLine', None)
                        ruleId = r['ruleId']
                        ruleIndex = r['ruleIndex']
                        rule_level = rules_metadata[ruleIndex]['defaultConfiguration']['level']
                        msg = r['message']['text']
                        if rule_level in ['error', 'warning']:
                            if rule is None or ruleId == rule:
                                codeql_results.append((uri, startLine, msg))
    return codeql_results


def parse_tagfile(tagfile):
    """
    Parses a ctags file to extract function definitions and their locations.

    Args:
        tagfile (str): The name of the ctags file to be parsed.

    Returns:
        dict: A dictionary where each key is a filename and the value is a list of tuples:
              (function name, start line number, end line number)
    """
    results = {}
    with open(full_path(tagfile)) as file:
        for line in file:
            line = line.rstrip()
            if line[0] == '!':
                continue

            startline, endline = 0, 0
            m = re.search(r"line:(\d+)", line)
            if m:
                startline = int(m.group(1))
            m = re.search(r"end:(\d+)", line)
            if m:
                endline = int(m.group(1))

            parts = line.split('\t')
            funcname = parts[0]
            filename = parts[1]

            results.setdefault(filename, []).append(
                (funcname, startline, endline))
    return results


def is_good(string):
    """
    Determines if a given string indicates a 'good' classification.

    Args:
        string (str): The string to be evaluated.

    Returns:
        bool: True if the string contains 'good' or 'Good', False otherwise.
    """
    return 'good' in string or 'Good' in string


def is_bad(string):
    """
    Determines if a given string indicates a 'bad' classification.

    Args:
        string (str): The string to be evaluated.

    Returns:
        bool: True if the string contains 'bad' or 'Bad', False otherwise.
    """
    return 'bad' in string or 'Bad' in string


class GroundTruth(Enum):
    """
    Enumeration representing the ground truth labels for CodeQL alerts.
    """
    GOOD = 1       # Indicates the alert is a false positive
    BAD = 2        # Indicates the alert is a true positive
    UNRELATED = 3  # Indicates the alert is unrelated
    ERR = 4        # Indicates an error in processing


def get_function_loc(tags, src_filename, lineno):
    """
    Identifies the function location that encompasses a specific line number within a source file.

    Args:
        tags (dict): A dictionary mapping filenames to lists of function definitions and their locations.
        src_filename (str): The name of the source file.
        lineno (int): The line number to be located within the source file.

    Returns:
        tuple or None: A tuple containing (function name, start line, end line) if a unique match is found;
                       None otherwise.
    """
    value = tags.get(src_filename)
    if value:
        matched_funcs = [
            (func_name, start, end) for func_name, start, end in value
            if start <= lineno <= end]
        if len(matched_funcs) == 1:
            return matched_funcs[0]
        else:
            print("Multiple or no match:", matched_funcs)
    else:
        print(src_filename, "not found in tags")
    return None


def get_ground_truth(src_filename, func_name):
    """
    Determines the ground truth label based on the source filename and function name.

    Args:
        src_filename (str): The name of the source file.
        func_name (str): The name of the function.

    Returns:
        GroundTruth: The corresponding ground truth enumeration value.
    """
    if is_good(src_filename) or is_good(func_name):
        return GroundTruth.GOOD
    if is_bad(src_filename) or is_bad(func_name):
        return GroundTruth.BAD
    return GroundTruth.UNRELATED


def dump_src(filename, start, end):
    """
    Extracts and returns a specific range of lines from a source file, optionally with line numbers.

    Args:
        filename (str): The name of the file to read from.
        start (int): The starting line number (1-based).
        end (int): The ending line number (inclusive, 1-based).

    Returns:
        str: A string containing the selected lines from the file, each prefixed with its line number.
    """
    with open(full_path(filename), 'r') as file:
        lines = file.readlines()

    # Ensure the line numbers are within the valid range
    start = max(1, start)  # Ensure start is at least 1
    end = min(len(lines), end)  # Ensure end does not exceed the number of lines

    # Slice the list to get lines from start to end (inclusive)
    selected_lines = lines[start-1:end]

    # Prepend line numbers
    numbered_lines = [f"{i+start}: {line}" for i, line in enumerate(selected_lines)]

    # Join the selected lines into a single string
    return ''.join(numbered_lines)


def build_dataset(tags, codeql_results):
    """
    Constructs a dataset by mapping CodeQL results to their corresponding function code and ground truth labels.

    Args:
        tags (dict): A dictionary mapping filenames to lists of function definitions and their locations.
        codeql_results (list): A list of tuples containing CodeQL alert details:
                               (URI of the source file, line number of the alert, alert message)

    Returns:
        list: A list of tuples, each containing:
              (line number, alert message, function code, ground truth label)
    """
    ds = []
    for uri, lineno, msg in codeql_results:
        func_loc = get_function_loc(tags, uri, lineno)
        if func_loc:
            func_name, func_startline, func_endline = func_loc
            gt = get_ground_truth(uri, func_name)
            func = dump_src(uri, func_startline, func_endline)
            ds.append((lineno, msg, func, gt))
    return ds


def main():
    """
    The main function orchestrates the parsing of CodeQL SARIF files, extraction of function definitions,
    determination of ground truth labels, and aggregation of classification metrics.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Process Juliet test suite SARIF files for CodeQL alerts classification."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory of the Juliet test suite repository."
    )
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = args.base_dir

    # Parse the ctags file to obtain function definitions
    tags = parse_tagfile('tags.txt')

    # Parse the CodeQL SARIF file to extract relevant results
    codeql_results = parse_codeql_sarif("cpp.sarif", "cpp/uninitialized-local")

    # Build the dataset by mapping CodeQL results to function code and ground truth labels
    ds = build_dataset(tags, codeql_results)

    # Initialize counters for classification metrics
    codeql_tp = 0
    codeql_fp = 0
    codeql_unrelated = 0

    # Iterate over the dataset to count true positives, false positives, and unrelated alerts
    for _, _, _, gt in ds:
        if gt == GroundTruth.GOOD:
            codeql_fp += 1
        elif gt == GroundTruth.BAD:
            codeql_tp += 1
        else:
            codeql_unrelated += 1

    # Print the classification metrics
    print("CodeQL Classification Metrics (just for un-initialized local variables warnings):")
    print(f"Number of True Positives (TP): {codeql_tp}")
    print(f"Number of False Positives (FP): {codeql_fp}")
    print(f"Number of Unrelated Alerts: {codeql_unrelated}")


if __name__ == '__main__':
    main()
