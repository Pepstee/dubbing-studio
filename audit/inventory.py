#!/usr/bin/env python3
"""Build deterministic forensic ledgers for the frozen repository corpus.

The audited corpus is every regular file below REPO except post-freeze audit/
artifacts. Symlinks are inventoried separately. No source file is modified.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import mimetypes
import os
import re
import stat
import subprocess
import tomllib
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "audit"

TEXT_SUFFIXES = {
    ".cfg", ".conf", ".css", ".csv", ".env", ".html", ".ini", ".jinja",
    ".jinja2", ".js", ".json", ".md", ".py", ".rst", ".service", ".sh",
    ".srt", ".svg", ".toml", ".ts", ".tsx", ".txt", ".yaml", ".yml",
}
LANGUAGES = {
    ".css": "CSS",
    ".html": "HTML",
    ".js": "JavaScript",
    ".json": "JSON",
    ".md": "Markdown",
    ".py": "Python",
    ".service": "systemd",
    ".sh": "Shell",
    ".srt": "SubRip",
    ".toml": "TOML",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".yaml": "YAML",
    ".yml": "YAML",
}
SECRET_NAME_TERMS = {
    ".env", "credential", "credentials", "id_rsa", "id_ed25519", "password",
    "passwd", "private_key", "secret", "token",
}
GENERATED_ROOTS = {
    ".git", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__",
    "graphify-out", "output",
}
VENDORED_ROOTS = {".venv"}


def run(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        args,
        cwd=REPO,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_path_sets() -> tuple[set[str], set[str], set[str], set[str]]:
    tracked = set(run("git", "ls-files", "-z").split("\0")) - {""}
    deleted = set(run("git", "ls-files", "--deleted", "-z").split("\0")) - {""}
    untracked = set(
        run("git", "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    ) - {""}
    ignored = set(
        run(
            "git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"
        ).split("\0")
    ) - {""}
    return tracked, deleted, untracked, ignored


def classify(path: Path, rel: str) -> tuple[str, str]:
    parts = Path(rel).parts
    suffix = path.suffix.lower()
    name = path.name.lower()
    first = parts[0] if parts else ""

    if first == ".git":
        return "git_internal", "generated"
    if first == ".venv":
        return "vendored_dependency", "generated"
    if first in {".pytest_cache", ".ruff_cache", "__pycache__"} or "__pycache__" in parts:
        return "cache", "generated"
    if first == "graphify-out":
        return "generated_knowledge_graph", "generated"
    if first == "output":
        return "generated_output", "generated"
    if first == "tests":
        if "fixtures" in parts or "support" in parts:
            return "test_fixture_or_helper", "human"
        return "test", "human"
    if first == "docs" or suffix in {".md", ".rst"}:
        return "documentation", "human"
    if first == "deploy":
        return "deployment", "human"
    if first == "schemas":
        return "schema", "human"
    if first == "samples":
        return "fixture", "human"
    if first == "dubbing":
        if suffix == ".py":
            return "source", "human"
        if "templates" in parts:
            return "template", "human"
        if "static" in parts:
            return "frontend_asset", "human"
        return "package_data", "human"
    if first.endswith(".egg-info"):
        return "generated_package_metadata", "generated"
    if name in {"pyproject.toml", "setup.cfg", "tox.ini", ".gitignore"}:
        return "configuration", "human"
    if rel == "acceptance" or suffix == ".py":
        return "script_or_source", "human"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".whl"}:
        return "archive", "generated"
    return "unknown", "human"


def file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pyc":
        return "Python bytecode"
    if suffix in {".so", ".dylib", ".dll"}:
        return "shared library"
    if suffix in {".zip", ".whl", ".gz", ".tar", ".bz2", ".xz"}:
        return "archive"
    mime = mimetypes.guess_type(path.name)[0]
    if suffix in TEXT_SUFFIXES or path.name in {
        ".gitignore", "HEAD", "config", "description", "packed-refs",
        "COMMIT_EDITMSG", "CACHEDIR.TAG", "RECORD", "METADATA", "WHEEL",
        "entry_points.txt", "top_level.txt",
    }:
        return mime or "text"
    try:
        head = path.open("rb").read(4096)
    except OSError:
        return "unreadable"
    if b"\0" in head:
        return mime or "binary"
    try:
        head.decode("utf-8")
        return mime or "text"
    except UnicodeDecodeError:
        return mime or "binary"


def full_read_candidate(rel: str, classification: str, ftype: str) -> bool:
    first = Path(rel).parts[0]
    if classification in {
        "source", "test", "test_fixture_or_helper", "documentation",
        "deployment", "schema", "fixture", "package_data", "template",
        "frontend_asset", "configuration", "script_or_source", "unknown",
    }:
        return (
            Path(rel).suffix.lower() in TEXT_SUFFIXES
            or ftype == "text"
            or ftype.startswith("text/")
            or "json" in ftype
        )
    if first == ".venv" and rel in {
        ".venv/pyvenv.cfg",
        ".venv/bin/activate",
        ".venv/bin/activate.csh",
        ".venv/bin/activate.fish",
    }:
        return True
    if rel in {".git/config", ".git/HEAD", ".git/info/exclude"}:
        return True
    if classification in {"generated_package_metadata", "generated_knowledge_graph"}:
        return Path(rel).name in {
            "PKG-INFO", "SOURCES.txt", "entry_points.txt", "requires.txt",
            "top_level.txt", "manifest.json",
        }
    return False


def structural_read_candidate(rel: str, ftype: str) -> bool:
    parts = Path(rel).parts
    if not parts or parts[0] != ".venv":
        return False
    if len(parts) >= 2 and parts[1] == "bin" and (
        ftype == "text" or ftype.startswith("text/")
    ):
        return True
    return any(part.endswith(".dist-info") for part in parts) and Path(rel).name in {
        "METADATA", "WHEEL", "entry_points.txt", "top_level.txt",
        "direct_url.json", "RECORD", "INSTALLER", "REQUESTED",
    }


class _HTMLStructure(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.start_tags = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.start_tags += 1


def parse_python(path: Path, text: str) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    tree = ast.parse(text, filename=str(path))
    symbols: list[dict[str, Any]] = []
    imports: list[str] = []
    calls: list[str] = []
    parents: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            qualified = ".".join([*parents, node.name])
            symbols.append(symbol_record(node, qualified, "class"))
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            qualified = ".".join([*parents, node.name])
            kind = "method" if parents else "function"
            symbols.append(symbol_record(node, qualified, kind))
            parents.append(node.name)
            self.generic_visit(node)
            parents.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Import(self, node: ast.Import) -> None:
            imports.extend(alias.name for alias in node.names)
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            imports.append(("." * node.level) + (node.module or ""))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            name = call_name(node.func)
            if name:
                calls.append(name)
            self.generic_visit(node)

    def symbol_record(node: ast.AST, qualified: str, kind: str) -> dict[str, Any]:
        decorators = []
        for dec in getattr(node, "decorator_list", []):
            decorators.append(call_name(dec) or ast.unparse(dec))
        return {
            "qualified_name": qualified,
            "kind": kind,
            "line": getattr(node, "lineno", 0),
            "end_line": getattr(node, "end_lineno", 0),
            "decorators": decorators,
        }

    def call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            left = call_name(node.value)
            return f"{left}.{node.attr}" if left else node.attr
        if isinstance(node, ast.Call):
            return call_name(node.func)
        return ""

    Visitor().visit(tree)
    return symbols, sorted(set(imports)), sorted(set(calls))


def purpose_for(rel: str, classification: str) -> str:
    path = Path(rel)
    if path.name == "__init__.py":
        return "Python package boundary and exported API"
    if path.name == "__main__.py":
        return "Python module execution entrypoint"
    if path.suffix == ".service":
        return "systemd service unit"
    if classification == "test":
        return "Automated behavioural or structural validation"
    if classification == "source":
        return f"Runtime module {'.'.join(path.with_suffix('').parts)}"
    if classification == "deployment":
        return "Deployment configuration or operational documentation"
    if classification == "schema":
        return "Machine-readable configuration/data contract"
    if classification == "documentation":
        return "Human-authored project documentation"
    if classification == "vendored_dependency":
        return "Local virtual-environment dependency or wrapper"
    if classification == "git_internal":
        return "Git repository metadata"
    if classification == "cache":
        return "Tool-generated cache"
    if classification.startswith("generated"):
        return classification.replace("_", " ")
    return classification.replace("_", " ")


def entrypoint_status(rel: str, symbols: list[dict[str, Any]], text: str | None) -> str:
    path = Path(rel)
    if path.name == "__main__.py":
        return "MODULE_ENTRYPOINT"
    if path.suffix == ".service":
        return "DEPLOYMENT_ENTRYPOINT"
    if rel == "acceptance":
        return "SHELL_WRAPPER_NONEXECUTABLE"
    if rel in {"pyproject.toml", "dubbing_studio.egg-info/entry_points.txt"}:
        return "ENTRYPOINT_REGISTRY"
    if text and any(s["qualified_name"] == "main" for s in symbols):
        return "POTENTIAL_MAIN"
    if text and ("@app.route" in text or "@app.get" in text or "@app.post" in text):
        return "WEB_ROUTE_CONTAINER"
    return "NOT_ENTRYPOINT"


def main() -> None:
    AUDIT.mkdir(exist_ok=True)
    tracked, deleted, untracked, ignored = git_path_sets()

    files = sorted(
        (
            path
            for path in REPO.rglob("*")
            if path.is_file() and not path.is_symlink()
            and (not path.relative_to(REPO).parts or path.relative_to(REPO).parts[0] != "audit")
        ),
        key=lambda p: os.fsencode(str(p.relative_to(REPO))),
    )
    symlinks = sorted(
        (
            path
            for path in REPO.rglob("*")
            if path.is_symlink()
            and (not path.relative_to(REPO).parts or path.relative_to(REPO).parts[0] != "audit")
        ),
        key=lambda p: os.fsencode(str(p.relative_to(REPO))),
    )

    preliminary: list[dict[str, Any]] = []
    python_data: dict[str, dict[str, Any]] = {}
    full_texts: dict[str, str] = {}

    for path in files:
        rel = path.relative_to(REPO).as_posix()
        classification, authorship = classify(path, rel)
        ftype = file_type(path)
        digest = sha256(path)
        language = LANGUAGES.get(path.suffix.lower(), "")
        read_status = "METADATA_ONLY"
        parser = "sha256+stat"
        reason_not_read = ""
        symbols: list[dict[str, Any]] = []
        imports: list[str] = []
        calls: list[str] = []
        text: str | None = None

        if full_read_candidate(rel, classification, ftype):
            try:
                text = path.read_text(encoding="utf-8")
                full_texts[rel] = text
                read_status = "FULL"
                parser = "utf8_full"
                if path.suffix.lower() == ".py":
                    symbols, imports, calls = parse_python(path, text)
                    parser = "utf8_full+python_ast"
                    python_data[rel] = {
                        "symbols": symbols,
                        "imports": imports,
                        "calls": calls,
                    }
                elif path.suffix.lower() == ".json":
                    json.loads(text)
                    parser = "utf8_full+json"
                elif path.suffix.lower() == ".toml":
                    tomllib.loads(text)
                    parser = "utf8_full+tomllib"
                elif path.suffix.lower() == ".service":
                    sections = [
                        line[1:-1]
                        for line in text.splitlines()
                        if line.startswith("[") and line.endswith("]")
                    ]
                    if not sections:
                        raise ValueError("systemd unit has no sections")
                    parser = "utf8_full+systemd_sections"
                elif path.suffix.lower() == ".html":
                    html_parser = _HTMLStructure()
                    html_parser.feed(text)
                    html_parser.close()
                    parser = f"utf8_full+html_parser:{html_parser.start_tags}_start_tags"
                elif path.suffix.lower() == ".css":
                    if text.count("{") != text.count("}"):
                        raise ValueError("unbalanced CSS blocks")
                    parser = "utf8_full+css_block_balance"
                elif path.suffix.lower() == ".srt":
                    blocks = [block for block in text.strip().split("\n\n") if block.strip()]
                    if not blocks or any("-->" not in block for block in blocks):
                        raise ValueError("malformed SRT block structure")
                    parser = f"utf8_full+srt_blocks:{len(blocks)}"
                elif rel == "acceptance":
                    subprocess.run(
                        ["bash", "-n", str(path)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    parser = "utf8_full+bash_n"
            except UnicodeDecodeError as exc:
                read_status = "UNREADABLE"
                reason_not_read = f"not valid UTF-8: {exc}"
            except (SyntaxError, json.JSONDecodeError) as exc:
                read_status = "FULL"
                reason_not_read = f"structural parser error: {exc}"
                parser += "+parse_failed"
            except OSError as exc:
                read_status = "UNREADABLE"
                reason_not_read = str(exc)
        elif structural_read_candidate(rel, ftype):
            try:
                text = path.read_text(encoding="utf-8")
                read_status = "STRUCTURAL"
                parser = "utf8_manifest_or_wrapper_structure"
                reason_not_read = (
                    "Vendored/generated manifest or wrapper inspected structurally; "
                    "not treated as first-party source."
                )
            except (OSError, UnicodeDecodeError) as exc:
                read_status = "UNREADABLE"
                reason_not_read = str(exc)
        else:
            reason_not_read = (
                "Generated, vendored, cached, Git-internal, or binary file; "
                "inventoried by metadata and digest per audit scope."
            )

        secret_risk = ""
        lower_rel = rel.lower()
        if any(term in lower_rel for term in SECRET_NAME_TERMS):
            secret_risk = "name suggests possible credential/secret material; values not reported"

        if rel in tracked:
            tracked_status = "TRACKED"
        elif rel.startswith(".git/"):
            tracked_status = "GIT_INTERNAL"
        elif rel in untracked:
            tracked_status = "UNTRACKED"
        else:
            tracked_status = "UNTRACKED_OR_GENERATED"

        if rel in ignored:
            ignored_status = "IGNORED"
        elif rel.startswith(".git/"):
            ignored_status = "GIT_INTERNAL"
        else:
            ignored_status = "NOT_IGNORED"

        preliminary.append(
            {
                "path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
                "tracked_status": tracked_status,
                "ignored_status": ignored_status,
                "file_type": ftype,
                "language": language,
                "classification": classification,
                "generated_or_human": authorship,
                "read_status": read_status,
                "parser_used": parser,
                "purpose": purpose_for(rel, classification),
                "principal_symbols": ";".join(s["qualified_name"] for s in symbols[:30]),
                "entrypoint_status": entrypoint_status(rel, symbols, text),
                "references_in": "",
                "references_out": ";".join(imports),
                "duplicate_group": "",
                "risk_notes": secret_risk,
                "reason_not_read": reason_not_read,
            }
        )

    digest_groups: dict[str, list[str]] = defaultdict(list)
    for row in preliminary:
        digest_groups[row["sha256"]].append(row["path"])
    duplicate_ids = {
        digest: f"SHA256-{index:04d}"
        for index, digest in enumerate(
            sorted(d for d, members in digest_groups.items() if len(members) > 1),
            start=1,
        )
    }

    module_to_rel: dict[str, str] = {}
    for rel in python_data:
        module = rel.removesuffix(".py").replace("/", ".")
        if module.endswith(".__init__"):
            module = module.removesuffix(".__init__")
        module_to_rel[module] = rel
    incoming: dict[str, set[str]] = defaultdict(set)
    for rel, data in python_data.items():
        for imported in data["imports"]:
            normalized = imported.lstrip(".")
            for module, target in module_to_rel.items():
                if normalized == module or normalized.startswith(module + "."):
                    incoming[target].add(rel)

    for row in preliminary:
        row["duplicate_group"] = duplicate_ids.get(row["sha256"], "")
        row["references_in"] = ";".join(sorted(incoming.get(row["path"], set())))

    ledger_fields = [
        "path", "size_bytes", "sha256", "tracked_status", "ignored_status",
        "file_type", "language", "classification", "generated_or_human",
        "read_status", "parser_used", "purpose", "principal_symbols",
        "entrypoint_status", "references_in", "references_out",
        "duplicate_group", "risk_notes", "reason_not_read",
    ]
    with (AUDIT / "FILE_LEDGER.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ledger_fields)
        writer.writeheader()
        writer.writerows(preliminary)

    symbol_rows: list[dict[str, Any]] = []
    for rel, data in sorted(python_data.items()):
        module = rel.removesuffix(".py").replace("/", ".")
        symbol_rows.append(
            {
                "path": rel,
                "qualified_name": module,
                "kind": "module",
                "line": 1,
                "end_line": full_texts[rel].count("\n") + 1,
                "decorators": "",
                "callers": ";".join(sorted(incoming.get(rel, set()))),
                "callees": ";".join(data["imports"]),
                "imports": ";".join(data["imports"]),
                "runtime_reachability": (
                    "STATICALLY_REFERENCED"
                    if incoming.get(rel)
                    else "EXTERNAL_OR_UNRESOLVED"
                ),
                "tests": "",
                "configuration_dependencies": "",
                "data_io": "",
                "side_effects": "",
                "external_services": "",
                "documented": "YES" if module in full_texts.get("README.md", "") else "NO_OR_UNKNOWN",
                "status": (
                    "TEST"
                    if rel.startswith("tests/")
                    else "ACTIVE_OR_EXTERNALLY_REACHABLE"
                ),
            }
        )
    for rel, data in sorted(python_data.items()):
        text = full_texts[rel]
        for symbol in data["symbols"]:
            name = symbol["qualified_name"]
            short_name = name.rsplit(".", 1)[-1]
            test_refs = sorted(
                candidate
                for candidate, candidate_text in full_texts.items()
                if candidate.startswith("tests/") and short_name in candidate_text
            )
            callers = sorted(
                f"{candidate}:{call}"
                for candidate, candidate_data in python_data.items()
                for call in candidate_data["calls"]
                if call == short_name or call.endswith("." + short_name)
            )
            decorator_text = ";".join(symbol["decorators"])
            kind = symbol["kind"]
            if any(
                marker in decorator_text
                for marker in (".route", ".get", ".post", ".errorhandler")
            ):
                kind = "web_route_or_handler"
            elif any(
                marker in decorator_text
                for marker in (".before_request", ".after_request")
            ):
                kind = "middleware"
            symbol_rows.append(
                {
                    "path": rel,
                    "qualified_name": name,
                    "kind": kind,
                    "line": symbol["line"],
                    "end_line": symbol["end_line"],
                    "decorators": decorator_text,
                    "callers": ";".join(callers),
                    "callees": ";".join(data["calls"] if symbol["kind"] == "function" and "." not in name else []),
                    "imports": ";".join(data["imports"]),
                    "runtime_reachability": "STATICALLY_REFERENCED" if callers else "UNRESOLVED_STATICALLY",
                    "tests": ";".join(test_refs),
                    "configuration_dependencies": "",
                    "data_io": "",
                    "side_effects": "",
                    "external_services": "",
                    "documented": "YES" if short_name in full_texts.get("README.md", "") else "NO_OR_UNKNOWN",
                    "status": (
                        "TEST"
                        if rel.startswith("tests/")
                        else "ACTIVE_OR_EXTERNALLY_REACHABLE"
                        if callers or kind in {"web_route_or_handler", "middleware"}
                        or short_name == "main"
                        else "AMBIGUOUS_NO_STATIC_CALLER"
                    ),
                }
            )

    def add_non_python_symbol(
        path: str,
        name: str,
        kind: str,
        *,
        reachability: str = "EXTERNALLY_REACHABLE",
        status: str = "ACTIVE",
        configuration: str = "",
        side_effects: str = "",
    ) -> None:
        symbol_rows.append(
            {
                "path": path,
                "qualified_name": name,
                "kind": kind,
                "line": 1,
                "end_line": 1,
                "decorators": "",
                "callers": "",
                "callees": "",
                "imports": "",
                "runtime_reachability": reachability,
                "tests": "",
                "configuration_dependencies": configuration,
                "data_io": "",
                "side_effects": side_effects,
                "external_services": "",
                "documented": "YES",
                "status": status,
            }
        )

    project_document = tomllib.loads(full_texts["pyproject.toml"])
    for command, target in sorted(project_document["project"]["scripts"].items()):
        add_non_python_symbol(
            "pyproject.toml",
            command,
            "console_command",
            configuration=target,
        )
    for command in sorted(
        set(
            re.findall(
                r"\.add_parser\(\s*[\"']([^\"']+)[\"']",
                full_texts["dubbing/cli/main.py"],
            )
        )
    ):
        add_non_python_symbol(
            "dubbing/cli/main.py",
            f"dubbing-cli {command}",
            "cli_subcommand",
        )
    for rel in sorted(
        path
        for path in full_texts
        if path.endswith(".service")
    ):
        add_non_python_symbol(
            rel,
            Path(rel).stem,
            "deployment_unit",
            side_effects="starts persistent local process",
        )
    for rel in sorted(
        path
        for path in full_texts
        if "/templates/" in path
    ):
        add_non_python_symbol(rel, rel, "template")
    add_non_python_symbol(
        "dubbing/apps/personal_capture/store.py",
        "captures",
        "database_table",
        reachability="RUNTIME_PERSISTENCE",
        side_effects="SQLite durable state",
    )
    add_non_python_symbol(
        "dubbing/apps/personal_capture/service.py",
        "giga.personal-capture-event.v1",
        "message_schema",
        reachability="FILE_BASED_EXTERNAL_INTERFACE",
        side_effects="writes reviewed evidence event",
    )
    add_non_python_symbol(
        "schemas/personal-capture-deployment.v1.schema.json",
        "dubbing.personal-capture-deployment.v1",
        "json_schema",
        reachability="DOCUMENTED_BUT_NOT_RUNTIME_LOADED",
        status="DORMANT_BUT_REACHABLE",
    )

    symbol_fields = [
        "path", "qualified_name", "kind", "line", "end_line", "decorators",
        "callers", "callees", "imports", "runtime_reachability", "tests",
        "configuration_dependencies", "data_io", "side_effects",
        "external_services", "documented", "status",
    ]
    with (AUDIT / "SYMBOL_LEDGER.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=symbol_fields)
        writer.writeheader()
        writer.writerows(symbol_rows)

    executable_files = []
    for path in files:
        mode = path.stat().st_mode
        if mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            executable_files.append(path.relative_to(REPO).as_posix())

    snapshot = {
        "root": str(REPO),
        "regular_file_count": len(files),
        "ledger_rows": len(preliminary),
        "symlinks": [
            {
                "path": path.relative_to(REPO).as_posix(),
                "target": os.readlink(path),
                "resolved": str(path.resolve(strict=False)),
                "target_exists": path.resolve(strict=False).exists(),
            }
            for path in symlinks
        ],
        "tracked_missing": sorted(deleted),
        "counts": {
            "read_status": Counter(row["read_status"] for row in preliminary),
            "tracked_status": Counter(row["tracked_status"] for row in preliminary),
            "ignored_status": Counter(row["ignored_status"] for row in preliminary),
            "classification": Counter(row["classification"] for row in preliminary),
            "generated_or_human": Counter(row["generated_or_human"] for row in preliminary),
            "language": Counter(row["language"] or "none" for row in preliminary),
        },
        "duplicate_groups": {
            duplicate_ids[digest]: members
            for digest, members in sorted(digest_groups.items())
            if digest in duplicate_ids
        },
        "python_modules": python_data,
        "executable_files": executable_files,
        "post_freeze_exclusion": "audit/",
    }
    (AUDIT / "inventory.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "regular_files": len(files),
        "ledger_rows": len(preliminary),
        "symbols": len(symbol_rows),
        "symlinks": len(symlinks),
        "read_status": snapshot["counts"]["read_status"],
        "classification": snapshot["counts"]["classification"],
    }, indent=2, default=dict))


if __name__ == "__main__":
    main()
