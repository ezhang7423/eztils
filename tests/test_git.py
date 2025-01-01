import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path
from eztils.git import Repo
from eztils.git import (
    copyanything,
    copy_git_repo,
    generate_snapshot,
    get_submodules,
    get_submodules_recursively,
    get_git_info,
)


@pytest.fixture
def temp_dir(tmp_path):
    d = tmp_path / "testdir"
    d.mkdir()
    return d


def test_copyanything(temp_dir):
    src_file = temp_dir / "source.txt"
    src_file.write_text("Hello")
    dst_file = temp_dir / "dest.txt"
    copyanything(str(src_file), str(dst_file))
    assert dst_file.exists()
    assert dst_file.read_text() == "Hello"


def test_copyanything_dir(temp_dir):
    src_dir = temp_dir / "srcdir"
    src_dir.mkdir()
    (src_dir / "file1.txt").write_text("File1")
    dst_dir = temp_dir / "dstdir"
    copyanything(str(src_dir), str(dst_dir))
    assert (dst_dir / "file1.txt").exists()
    assert (dst_dir / "file1.txt").read_text() == "File1"


@patch("subprocess.run")
def test_copy_git_repo(mock_run, temp_dir):
    mock_run.return_value.stdout = "file1.txt\nfolder/file2.txt\n"
    root = str(temp_dir / "repo")
    os.makedirs(root, exist_ok=True)
    (temp_dir / "repo" / "file1.txt").write_text("test1")
    folder = temp_dir / "repo" / "folder"
    folder.mkdir(parents=True)
    (folder / "file2.txt").write_text("test2")
    dest = temp_dir / "dest"
    copy_git_repo(root, str(dest))
    assert (dest / "file1.txt").read_text() == "test1"
    assert (dest / "folder" / "file2.txt").read_text() == "test2"


@patch("subprocess.run")
@patch("git.Repo")
def test_generate_snapshot(mock_repo_class, mock_run, temp_dir):
    mock_run.return_value.stdout = "file1.txt\n"
    repo_instance = MagicMock()
    repo_instance.working_tree_dir = str(temp_dir / "repo")
    repo_instance.submodules = []
    head_commit = MagicMock()
    head_commit.hexsha = "abc123"
    repo_instance.head.commit = head_commit
    repo_instance.active_branch.name = "main"
    mock_repo_class.return_value = repo_instance

    generate_snapshot(str(temp_dir / "repo"), str(temp_dir / "snap"))
    git_info_file = temp_dir / "snap" / "git_infos.txt"
    assert git_info_file.exists()
    content = git_info_file.read_text()
    assert "abc123" in content


@patch("git.Repo")
def test_get_submodules(mock_repo_class, temp_dir):
    mock_submodule = MagicMock()
    mock_submodule.path = "submod"
    mock_repo = MagicMock()
    mock_repo.working_tree_dir = str(temp_dir)
    mock_repo.submodules = [mock_submodule]
    mock_repo_class.return_value = mock_repo
    subs = get_submodules(mock_repo)
    assert len(subs) == 1


@patch("git.Repo")
def test_get_submodules_recursively(mock_repo_class, temp_dir):
    mock_submodule1 = MagicMock()
    mock_submodule1.path = "mod1"
    mock_submodule1.submodules = []
    mock_submodule2 = MagicMock()
    mock_submodule2.path = "mod2"
    mock_submodule2.submodules = []
    mock_repo = MagicMock()
    mock_repo.working_tree_dir = str(temp_dir)
    mock_repo.submodules = [mock_submodule1, mock_submodule2]
    mock_repo_class.return_value = mock_repo
    subs = get_submodules_recursively(mock_repo)
    assert len(subs) == 2


@patch("git.Repo")
def test_get_git_info(mock_repo_class, temp_dir):
    mock_repo = MagicMock()
    mock_repo.git.diff.side_effect = ["diff_unstaged", "diff_staged"]
    mock_repo.head.commit.hexsha = "1234abcd"
    mock_repo.active_branch.name = "main"
    mock_repo_class.return_value = mock_repo
    info = get_git_info(str(temp_dir))
    assert info.directory == str(temp_dir)
    assert info.commit_hash == "1234abcd"
    assert info.branch_name == "main"
