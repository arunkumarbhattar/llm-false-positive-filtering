import os
import subprocess
import dataset
import sys

if __name__ == '__main__':
    repo_path = "/home/mjshen/IVOS/repos/mbed-os-example-blinky"
    sarif_path = "/home/mjshen/IVOS/OSSEmbeddedResults/IVOES/mbed-os-example-blinky/43d51eaa7738b716de6222083d48bb21e249f6c6/cpp.sarif"
    new_sarif_path = "/home/mjshen/IVOS/OSSEmbeddedResults/IVOES/mbed-os-example-blinky/43d51eaa7738b716de6222083d48bb21e249f6c6/cpp-labeled.sarif"
    ds = dataset.Dataset(repo_path, sarif_path)
    ds.manually_label_codeql_results(new_sarif_path)
    sys.exit(0)

    SARIF_DIR = "/home/mjshen/IVOS/OSSEmbeddedResults"
    REPO_DIR = "/home/mjshen/IVOS/repos"

    with open('repos.txt', 'r') as f:
        repos = f.readlines()

    for repo in repos:
        print(f"\033[1;34m{repo}\033[0m")
        owner, repo_name = repo.strip().split('/')
        sarif_repo_dir = os.path.join(SARIF_DIR, owner, repo_name)

        if not os.path.exists(sarif_repo_dir):
            print(f"{sarif_repo_dir} not exists")
            continue
        for commit in os.listdir(sarif_repo_dir):
            sarif_repo_commit_path = os.path.join(sarif_repo_dir, commit)
            if not os.path.isdir(sarif_repo_commit_path):
                print(f"{sarif_repo_commit_path} is not a dir")
                continue
            sarif_path = os.path.join(sarif_repo_commit_path, 'cpp.sarif')
            new_sarif_path = os.path.join(sarif_repo_commit_path, 'cpp-labeled.sarif')
            if os.path.exists(new_sarif_path):
                print(f"{new_sarif_path} already exists")
                continue
            repo_path = os.path.join(REPO_DIR, repo_name)
            if not os.path.isdir(repo_path):
                print(f"{repo_path} is not a dir")
                continue
            subprocess.run(['git', 'checkout', commit], cwd=repo_path, check=True)
            ds = dataset.Dataset(repo_path, sarif_path)
            ds.manually_label_codeql_results(new_sarif_path)
