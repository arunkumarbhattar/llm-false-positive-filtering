# Juliet Test Suite Analysis Toolkit

## Overview

The Juliet Test Suite Analysis Toolkit is a collection of scripts designed to facilitate the analysis of the Juliet Test Suite using CodeQL, a powerful static analysis tool. This toolkit automates the process of compiling test cases into LLVM bitcode, analyzing them with CodeQL to generate SARIF (Static Analysis Results Interchange Format) files, labeling findings with ground truth data, and building comprehensive datasets for evaluation purposes.

# Table of Contents

    Overview
    Prerequisites
    Installation
    Scripts Overview
        1. add_ground_truth.py
        2. parse.py
        3. dataset_juliet.py
    Usage
    Additional Scripts
    Workflow Example

# Prerequisites
Before using the toolkit, ensure that the following tools and dependencies are installed:


    Python 3.6+
    CodeQL: Install from GitHub CodeQL
    Clang: For compiling test cases into LLVM bitcode.
    wllvm: Whole-program LLVM compiler wrappers (wllvm and wllvm++).
    ctags: For generating tags files.

# Installation

Download the juliet repo 

