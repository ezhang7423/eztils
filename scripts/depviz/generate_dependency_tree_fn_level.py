from itertools import chain, islice
import os
import ast
import graphviz
import importlib.util
from isort import place_module


def get_imports(file_path):
    """
    Parse a Python file's AST to find imported modules.
    """
    imports = []
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    for node in ast.walk(tree):
        # Handle "import x" or "import x as y" # todo DOUBLE CHECK HOW THE as gets handled
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        # Handle "from x import y" or "from x import y, z as w"
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append(node.module)
                for alias in node.names:
                    # module_name = f"{node.module}.{alias.name}"
                    # imports.append(module_name)
                    imports.append((node.module, alias.name))
            else:
                # Relative import (e.g. from . import something)
                # can be handled or ignored, depending on your needs.
                # TODO figure out how to handle this
                pass

    return list(set(imports))


def get_functions_and_calls(file_path):
    """
    Parse the AST for a Python file and return:
      1) The list of function definitions.
      2) A mapping from each function definition to the set of called function names (best-effort).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        file_contents = f.read()

    tree = ast.parse(file_contents, filename=file_path)

    functions = []  # List of function names defined in this file
    calls_in_function = {}
    current_function_stack = []

    class FuncCallVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            func_name = node.name
            current_function_stack.append(func_name)

            if func_name not in calls_in_function:
                calls_in_function[func_name] = set()
            functions.append(func_name)

            self.generic_visit(node)
            current_function_stack.pop()

        def visit_AsyncFunctionDef(self, node):
            # Treat async defs like normal function defs
            self.visit_FunctionDef(node)

        def visit_Call(self, node):
            if current_function_stack:
                caller_func = current_function_stack[-1]
                func_name = ""

                # Attempt to extract the called function name
                if isinstance(node.func, ast.Name):
                    # e.g. foo(...)
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    # e.g. module.submodule.foo(...)
                    parts = []
                    current_attr = node.func
                    while isinstance(current_attr, ast.Attribute):
                        parts.append(current_attr.attr)
                        current_attr = current_attr.value
                    if isinstance(current_attr, ast.Name):
                        parts.append(current_attr.id)
                    func_name = ".".join(parts[::-1])

                if func_name:
                    calls_in_function[caller_func].add(func_name)

            self.generic_visit(node)

    visitor = FuncCallVisitor()
    visitor.visit(tree)

    return functions, calls_in_function


def find_local_file_for_module(module_name, root_dir):
    """
    Attempt to resolve a Python module import to an actual .py file
    within the root directory (if it exists).
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if (
            spec
            and spec.origin
            and spec.origin.endswith(".py")
            and root_dir in spec.origin
        ):
            return os.path.relpath(spec.origin, root_dir)
    except (ModuleNotFoundError, ValueError):
        pass

    return None


def build_file_clusters(root_dir, with_external_deps=False):
    """
    Build a data structure describing each .py file’s functions, calls, and imports.
    """
    data = {"files": {}}

    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(subdir, filename)
                rel_path = os.path.relpath(file_path, root_dir)

                funcs, calls = get_functions_and_calls(file_path)
                imports = get_imports(file_path)

                data["files"][rel_path] = {
                    "functions": funcs,
                    "calls": calls,  # dict of {function_name: set_of_called_names}
                    "imports": set(imports),
                }

    return data


def sanitize_cluster_name(file_path):
    # Replace or remove characters that can break cluster naming
    return "cluster_" + file_path.replace("\\", "_").replace("/", "_").replace(
        ":", "_"
    ).replace(".", "_")


def sanitize_id(file_path, func_name):
    """
    Generate a Graphviz-safe node ID by removing or replacing
    problematic characters (especially the colon).
    """
    safe_path = (
        file_path.replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "_")
    )
    return f"{safe_path}__{func_name}"


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def create_dependency_graph(
    root_dir, with_external_deps=False, max_clusters_per_rank=100
):
    """
    Create a dependency graph at the function level. Each file is a cluster,
    each function is a node. If function A in file1 calls function B in file2,
    draw an edge. Optionally display external calls.
    """
    dot = graphviz.Digraph(comment="Function-Level Dependency Tree")
    dot.attr(
        # rankdir="TB",
        # ratio="compress",
        ratio='fill',
        size='16,9!',
        nodesep="0.75",
        ranksep="0.75",
        pack="true",
        overlap="scale",
        packmode="clust",
        newrank="true",
        # sep='10',
        # esep='9',
    )
    dot.attr("graph", splines="ortho")
    dot.attr("node", shape="rectangle")

    # use fdp for better layout
    dot.attr("graph", layout="fdp")
    # dot.attr("graph", layout="neato")
    # dot.attr("graph", layout="sfdp")

    data = build_file_clusters(root_dir, with_external_deps)
    external_calls = set()

    for file_path, file_info in data["files"].items():
        cluster_name = sanitize_cluster_name(file_path)
        with dot.subgraph(name=cluster_name) as sub:

            sub.attr(label=file_path, style="filled", color="lightgrey")

            for func_name in file_info["functions"]:
                node_id = sanitize_id(file_path, func_name)
                sub.node(node_id, label=func_name)

            # add invisible node to referene the cluster
            sub.node(f"{cluster_name}_node", style="invis", width="0", height="0")
   
    # Build a lookup of file -> set_of_function_names
    file_to_functions = {
        fp: set(info["functions"]) for fp, info in data["files"].items()
    }

    # Attempt to map imported modules to local files (or None if external)
    module_resolutions = {}
    for fp, file_info in data["files"].items():
        for mod in file_info["imports"]:
            if not mod:
                continue

            fn_name = None  # if it's a from import, we need to know the function/module name. so this is the 'y' part of from x import y
            if len(mod) == 2:
                # this is a from import
                fn_name = mod[1]
                mod = f"{mod[0]}.{fn_name}"

            # If it's standard library and we don't want external deps, skip it
            if place_module(mod.split(".")[0]) == "STDLIB" and not with_external_deps:
                continue

            local_file = find_local_file_for_module(mod, root_dir)
            if local_file:
                if fn_name:
                    module_resolutions[fn_name] = local_file
                else:
                    module_resolutions[mod] = local_file

            else:
                # Maybe only the top-level is local, e.g. "my_pkg"
                # vs "my_pkg.submod"
                # TODO double check this. this is probably wrong
                top_level = mod.split(".")[0]
                local_file = find_local_file_for_module(top_level, root_dir)
                if local_file:
                    module_resolutions[top_level] = local_file
                else:
                    # It's truly external or unresolvable
                    module_resolutions[mod] = None

    # Create edges for calls
    for file_path, file_info in data["files"].items():
        # if 'prompts.py' in file_path:
        # breakpoint()        
        # pass
        depends_on_files = []
        for caller_func, called_set in file_info["calls"].items():
            # if caller_func == "upload_scraped_data":
            # breakpoint()
            caller_id = sanitize_id(file_path, caller_func)

            for called_func_name in called_set:
                # 1) Check if it's a function in the same file
                if called_func_name in file_to_functions[file_path]:
                    callee_id = sanitize_id(file_path, called_func_name)
                    add_edge_if_not_exists(dot, caller_id, callee_id)
                    continue

                # 2) Possibly "module.submodule.func"
                parts = called_func_name.split(".")
                found_local = False
                # Try progressively shorter module names:
                # e.g. for "a.b.c", try "a.b" => "c", then "a" => "b.c"
                for i in range(len(parts) - 1, 0, -1):
                    mod_candidate = ".".join(parts[:i])
                    func_candidate = ".".join(parts[i:])
                    # if mod_candidate == 'db_utils' and func_candidate == 'upload_data_structure_to_s3':
                    # breakpoint()
                    if mod_candidate in module_resolutions:
                        local_file = module_resolutions[mod_candidate]
                        if (
                            local_file is not None
                            and func_candidate in file_to_functions.get(local_file, [])
                        ):
                            depends_on_files.append(local_file)                            
                            add_edge_if_not_exists(
                                dot,
                                caller_id,
                                sanitize_id(local_file, func_candidate),
                            )
                            found_local = True
                            break

                # If we didn't find a local match, consider it external/unresolved
                if not found_local:
                    if with_external_deps:
                        external_calls.add(called_func_name)
                        ext_node_id = f"EXTERNAL__{called_func_name}"
                        add_edge_if_not_exists(dot, caller_id, ext_node_id)

        # if not (file_info["imports"] and file_info["calls"] == {}):
        # continue

        # print(file_path, depends_on_files)
        for import_ in file_info["imports"]:
            if import_:
                if len(import_) == 2:
                    import_ = import_[1]
                if import_ in module_resolutions:
                    local_file = module_resolutions[import_]
                    if local_file and local_file not in depends_on_files:                            
                            add_edge_if_not_exists(
                                dot,
                                sanitize_cluster_name(file_path) + "_node",
                                sanitize_cluster_name(local_file) + "_node",
                                style="dashed",
                            )
                    elif with_external_deps:
                        external_calls.add(import_)
                        ext_node_id = f"EXTERNAL__{import_}"
                        add_edge_if_not_exists(
                            dot,
                            sanitize_cluster_name(file_path) + "_node",
                            ext_node_id,
                            style="dashed",
                        )

    # Make a cluster for external calls (optional)
    if with_external_deps and external_calls:
        with dot.subgraph(name="cluster_EXTERNAL") as sub:
            sub.attr(label="EXTERNAL", style="filled", color="#ffeeee")
            for external_func in external_calls:
                ext_node_id = f"EXTERNAL__{external_func}"
                sub.node(ext_node_id, external_func)

    return dot


def add_edge_if_not_exists_creator():
    added_edges = set()

    def add_edge_if_not_exists(dot, caller_id, callee_id, **kwargs):
        if not (caller_id, callee_id) in added_edges:
            dot.edge(caller_id, callee_id, **kwargs, penwidth=".5", weight='.5')
            added_edges.add((caller_id, callee_id))

    return add_edge_if_not_exists


add_edge_if_not_exists = add_edge_if_not_exists_creator()

if __name__ == "__main__":
    WITH_EXTERNAL_DEPENDENCIES = False
    root_directory = "llm_forecasting"  # change to your project folder
    dependency_graph = create_dependency_graph(
        root_directory, with_external_deps=WITH_EXTERNAL_DEPENDENCIES
    )
    dependency_graph.render(
        "dependency_tree_fn_level_no_external", format="svg", view=True
    )
