import pytest
import tempfile
import os
from pathlib import Path

from scripts.depviz.generate_dependency_tree_fn_level import (
    get_imports,
    get_functions_and_calls,
    find_local_file_for_module,
    build_file_clusters,
    create_dependency_graph,
)


def test_get_imports_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_code = (
            "import math\nimport os.path\nfrom collections import defaultdict\n"
        )
        test_file = Path(tmpdir) / "sample.py"
        test_file.write_text(sample_code, encoding="utf-8")

        imports = get_imports(str(test_file))
        assert "math" in imports
        assert "os.path" in imports
        # 'collections' appears plus ('collections','defaultdict') tuple from "from import"
        assert any(
            isinstance(x, tuple) and x[0] == "collections" and x[1] == "defaultdict"
            for x in imports
        )


def test_get_functions_and_calls():
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_code = """
def foo():
    bar()
    return 1

def bar():
    return 2

def baz():
    print(foo())
"""
        test_file = Path(tmpdir) / "functions.py"
        test_file.write_text(sample_code, encoding="utf-8")

        funcs, calls = get_functions_and_calls(str(test_file))
        assert set(funcs) == {"foo", "bar", "baz"}
        assert calls["foo"] == {"bar"}
        assert calls["baz"] == {"print", "foo"}


def test_find_local_file_for_module():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "dummy_module.py").write_text(
            "def hello(): pass", encoding="utf-8"
        )
        result = find_local_file_for_module("dummy_module", tmpdir)
        assert result is not None
        assert "dummy_module.py" in result


def test_build_file_clusters():
    with tempfile.TemporaryDirectory() as tmpdir:
        code_a = "import math\ndef func_a(): pass"
        code_b = "from math import sqrt\ndef func_b(): pass"
        file_a = Path(tmpdir) / "a.py"
        file_b = Path(tmpdir) / "b.py"
        file_a.write_text(code_a, encoding="utf-8")
        file_b.write_text(code_b, encoding="utf-8")

        data = build_file_clusters(tmpdir)
        assert len(data["files"]) == 2
        assert "a.py" in data["files"]
        assert "b.py" in data["files"]
        assert "func_a" in data["files"]["a.py"]["functions"]
        assert "func_b" in data["files"]["b.py"]["functions"]
        assert "math" in data["files"]["a.py"]["imports"]
        # 'math', ('math','sqrt') for b
        assert any(
            isinstance(x, tuple) and x[0] == "math" and x[1] == "sqrt"
            for x in data["files"]["b.py"]["imports"]
        )


def test_create_dependency_graph():
    with tempfile.TemporaryDirectory() as tmpdir:
        code_main = """

def main():
    helper()
"""
        code_other = """
def helper():
    pass
"""
        (Path(tmpdir) / "main.py").write_text(code_main, encoding="utf-8")
        (Path(tmpdir) / "other_module.py").write_text(code_other, encoding="utf-8")

        dot = create_dependency_graph(tmpdir, with_external_deps=False)
        dot_str = dot.source
        assert "main.py" in dot_str
        assert "cluster_main_py" in dot_str
        assert "other_module.py" in dot_str
        assert "cluster_other_module_py" in dot_str
        # check an edge from main->helper
        assert "main_py__main" in dot_str
        assert "other_module_py__helper" in dot_str
