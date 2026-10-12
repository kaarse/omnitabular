#!/usr/bin/env python3
"""Generate website API data from source without importing ML dependencies.

Run python scripts/build_docs.py; use --check in CI to detect stale reference data.
The page layout and guides remain hand-written and are never overwritten.
"""
import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "website/src/data/api-reference.json"
SOURCES = {"OmniTabular": "omnitabular", "HillClimbingEnsemble": "hill_climbing", "WeightedEnsemble": "ensemble"}


def parameters(function):
    args = function.args
    positional = args.posonlyargs + args.args
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    entries = []
    for arg, default in zip(positional + args.kwonlyargs, defaults + args.kw_defaults):
        if arg.arg in {"self", "cls"}:
            continue
        entries.append({"name": arg.arg, "type": ast.unparse(arg.annotation) if arg.annotation else "array-like",
                        "default": ast.unparse(default) if default is not None else None})
    return entries


def generate():
    result = {}
    for name, module in SOURCES.items():
        tree = ast.parse((ROOT / f"omnitabular/{module}.py").read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
        functions = [node for node in cls.body if isinstance(node, ast.FunctionDef)]
        init = next(node for node in functions if node.name == "__init__")
        result[name] = {"parameters": parameters(init), "methods": [
            {"name": node.name, "parameters": parameters(node), "description": (ast.get_docstring(node) or "").split("\n")[0]}
            for node in functions if not node.name.startswith("_")
        ]}
        if name == "OmniTabular":
            registry = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "_MODEL_REGISTRY" for target in node.targets))
            result[name]["models"] = [ast.literal_eval(key) for key in registry.keys]
    return json.dumps(result, indent=2, ensure_ascii=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != content:
            raise SystemExit("API data is stale. Run python scripts/build_docs.py and commit the result.")
        print("API reference data is up to date.")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(content)
        print(f"Generated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
