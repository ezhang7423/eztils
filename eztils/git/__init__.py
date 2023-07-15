import os
import pickle
import subprocess
import sys
from collections import namedtuple

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


def generate_snapshot(root, dest_path):
    """Given a destination path, copy the local root dir to the path. To
    save disk space, only ``*.py`` files will be copied.
    This function can be used to generate a snapshot of the repo so that the
    exactly same code status will be recovered when later playing a trained
    model or launching a grid-search job in the waiting queue.
    Args:
        root (str): the path to the repo
        dest_path (str): the path to generate a snapshot of repo
    """

    def rsync(src, target, includes, excludes):
        args = ["rsync", "-rI"]
        args += ["--exclude=%s" % i for i in excludes]
        args += ["--include=%s" % i for i in includes]
        args += [src, target]
        # shell=True preserves string arguments
        subprocess.check_call(
            " ".join(args), stdout=sys.stdout, stderr=sys.stdout, shell=True
        )

    with open(os.path.join(root, ".gitignore")) as fin:
        excludes = fin.read().splitlines()

    # these files are important for code status
    includes = ["*"]
    rsync(root, dest_path, includes, excludes)


def get_git_infos(dirs):
    try:
        import git

        git_infos = []
        for directory in dirs:
            # Idk how to query these things, so I'm just doing try-catch
            try:
                repo = git.Repo(directory)
                try:
                    branch_name = repo.active_branch.name
                except TypeError:
                    branch_name = "[DETACHED]"
                git_infos.append(
                    GitInfo(
                        directory=directory,
                        code_diff=repo.git.diff(None),
                        code_diff_staged=repo.git.diff("--staged"),
                        commit_hash=repo.head.commit.hexsha,
                        branch_name=branch_name,
                    )
                )
            except git.exc.InvalidGitRepositoryError as e:
                print(f"Not a valid git repo: {directory}")
    except ImportError:
        git_infos = None
    return git_infos
