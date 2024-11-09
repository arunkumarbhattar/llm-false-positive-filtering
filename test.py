import subprocess
import os

def test_juliet(juliet_root):
    def test_subdir(subdir):
        sarif_path = os.path.join(juliet_root, subdir, "cpp-labeled.sarif")
        bitcode_path = os.path.join(juliet_root, subdir, "partial.o.bc")
        subprocess.run(['python3', '-u', 'llm_triage.py', juliet_root, sarif_path, bitcode_path])
    #test_subdir("C/testcases/CWE242_Use_of_Inherently_Dangerous_Function/")
    #test_subdir("C/testcases/CWE457_Use_of_Uninitialized_Variable/s01/")
    test_subdir("C/testcases/CWE457_Use_of_Uninitialized_Variable/s02/")
    test_subdir("C/testcases/CWE121_Stack_Based_Buffer_Overflow/s01/")
    #for i in range(1, 10):
    #    test_subdir(f"C/testcases/CWE121_Stack_Based_Buffer_Overflow/s{i:02}/")
    #for i in range(1, 12):
    #    test_subdir(f"C/testcases/CWE122_Heap_Based_Buffer_Overflow/s{i:02}/")

test_juliet("/home/arun/Downloads/juliet-test-suite-for-c-cplusplus-v1-3")
