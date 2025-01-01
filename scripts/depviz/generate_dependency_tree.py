import os
import graphviz
import importlib.util
from isort import place_module
import ast

# Removed unused imports: distutils, glob, pkgutil, sys
# Removed duplicate import of os


def get_imports(file_path):
    """
    Parse a Python file and return a deduplicated list of imported modules (dotted paths).
    """
    imports = []
    # Specify UTF-8 encoding to avoid issues with non-ASCII content
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=file_path)

    for node in ast.walk(tree):
        # Handle "import x" or "import x as y"
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        # Handle "from x import y" or "from x import y, z as w"
        elif isinstance(node, ast.ImportFrom):
            # node.module can be None if it's a relative import like 'from . import something'
            if node.module is not None:
                imports.append(node.module)

            for alias in node.names:
                # e.g. "x.y"                
                module_name = f"{node.module}.{alias.name}"
                imports.append(module_name)

    return list(set(imports))


def create_dependency_graph(root_dir, with_external_deps=False):
    """
    Creates a Graphviz Digraph representing import dependencies for each .py file in root_dir.
    If with_external_deps is True, edges to standard lib or site-packages modules are also drawn.
    """
    dot = graphviz.Digraph(comment="Import Dependency Tree")
    dot.attr(
        rankdir="LR", ratio="fill", nodesep="0.3", ranksep="0.5", dpi="300"
    )  # Compact layout settings
    # dot.engine = 'fdp'
    dot.attr("graph", splines="ortho")
    dot.attr("node", shape="rectangle")  # Set node shape to rectangle

    for subdir, _, files in os.walk(root_dir):
        for filename in files:
            if filename.endswith(".py"):
                file_path = os.path.join(subdir, filename)
                imports = get_imports(file_path)
                node = os.path.relpath(file_path, root_dir)
                dot.node(node, node)

                for imp in imports:
                    # Check if it's standard library
                    if place_module(imp) == "STDLIB":
                        if with_external_deps:
                            dot.edge(node, imp)
                        continue

                    # Attempt to locate the module within the project
                    try:
                        spec = importlib.util.find_spec(imp)
                        if spec and spec.origin and spec.origin.endswith(".py"):
                            imp_path = os.path.relpath(spec.origin, root_dir)
                            if "site-packages" in spec.origin:
                                # It's an external file in site-packages
                                if with_external_deps:
                                    dot.edge(
                                        node, imp
                                    )  # link to external module by name
                            else:
                                # It's likely part of our local project
                                dot.edge(node, imp_path)
                    except ModuleNotFoundError:
                        # If we can't resolve it at all, decide whether to show it
                        if with_external_deps:
                            dot.edge(node, imp)

    return dot


if __name__ == "__main__":
    # Adjust this as needed
    WITH_EXTERNAL_DEPENDENCIES = False

    root_directory = "llm_forecasting"  # Change this to your root directory
    dependency_graph = create_dependency_graph(
        root_directory, with_external_deps=WITH_EXTERNAL_DEPENDENCIES
    )
    dependency_graph.render("dependency_tree", format="png", view=True)
