import errno
import os
import pickle
import shutil
import subprocess
import sys
from collections import namedtuple
from pathlib import Path

from git import Repo

GitInfo = namedtuple(
    "GitInfo",
    [
        "directory",
        "code_diff",
        "code_diff_staged",
        "commit_hash",
        "branch_name",
    ],
)


def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc:  # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else:
            raise


def copy_git_repo(root: str, dest_path: str):
    """Given a destination path, copy the local root dir to the path.
    This function can be used to generate a snapshot of the repo so that the
    exactly same code status will be recovered when later playing a trained
    model or launching a grid-search job in the waiting queue.
    Args:
        root (str): the path to the repo
        dest_path (str): the path to generate a snapshot of repo
    """
    root_path = Path(root)
    dest_path = Path(dest_path)
    # Get output from git ls-files --others --exclude-standard, split into list
    files = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=root_path,
        capture_output=True,
        text=True,
    ).stdout.splitlines()

    # Create the destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)

    # Copy files to destination, preserving directory structure
    for src_file in files:
        src_file = Path(src_file)
        dest_file = dest_path / src_file
        # Create necessary directories in destination
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        src = root_path / src_file
        if not src.exists():
            continue  # ignore deleted files that were listed by git ls-files

        # Copy file from source to destination
        copyanything(root_path / src_file, dest_file)

    copyanything(root_path / ".git", dest_path / ".git")

    # Recursively process submodules
    root_repo = Repo(root)
    submodules = get_submodules(root_repo)

    for submodule in submodules:
        # Call copy_git_repo recursively on the submodule
        copy_git_repo(
            submodule.working_tree_dir,
            dest_path / Path(submodule.working_tree_dir).relative_to(root_path),
        )


def generate_snapshot(root: str, dest_path: str):
    copy_git_repo(root, dest_path)
    dest_path = Path(dest_path)

    root_repo = Repo(root)
    submodules = get_submodules_recursively(root_repo)

    git_infos = [get_git_info(i.working_tree_dir) for i in [root_repo] + submodules]
    for directory, code_diff, code_diff_staged, commit_hash, branch_name in git_infos:
        directory_path = Path(directory)
        diff_file_name = (directory_path.absolute().as_posix()[1:] + ".patch").replace(
            "/", "-"
        )
        diff_staged_file_name = (
            directory_path.absolute().as_posix()[1:] + "_staged.patch"
        ).replace("/", "-")

        if code_diff is not None and len(code_diff) > 0:
            with open(dest_path / diff_file_name, "w") as f:
                f.write(code_diff + "\n")
        if code_diff_staged is not None and len(code_diff_staged) > 0:
            with open(dest_path / diff_staged_file_name, "w") as f:
                f.write(code_diff_staged + "\n")

        with open(dest_path / "git_infos.txt", "a") as f:
            f.write(f"directory: {directory}\n")
            f.write(f"git hash: {commit_hash}\n")
            f.write(f"git branch name: {branch_name}\n\n")


def get_submodules_recursively(root: Repo):
    ret = []
    for i in get_submodules(root):
        ret += [i] + get_submodules_recursively(i)
    return ret


def get_submodules(root: Repo):
    return [Repo(Path(root.working_tree_dir) / i.path) for i in root.submodules]


def get_git_info(dir):
    repo = Repo(dir)
    try:
        branch_name = repo.active_branch.name
    except TypeError:
        branch_name = "[DETACHED]"
    return GitInfo(
        directory=dir,
        code_diff=repo.git.diff(None),
        code_diff_staged=repo.git.diff("--staged"),
        commit_hash=repo.head.commit.hexsha,
        branch_name=branch_name,
    )
