"""Adversarial tests for validation.mutation._shipped_sources package-only filter.

Key invariant under test: _shipped_sources excludes any .py file whose parent directory
lacks an __init__.py — i.e. top-level scripts such as acceptance.py are never mutated,
only code inside proper packages.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from validation.mutation import _shipped_sources, run_mutation_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names(paths: list[Path]) -> list[str]:
    return [p.name for p in paths]


def _write_package(root: Path, pkg_name: str, *module_names: str) -> Path:
    pkg = root / pkg_name
    pkg.mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("")
    for name in module_names:
        (pkg / name).write_text("x = 1\n")
    return pkg


# ---------------------------------------------------------------------------
# Core: top-level scripts excluded
# ---------------------------------------------------------------------------

class TestTopLevelScriptsExcluded:
    """Files at project root (no __init__.py in that directory) must be absent."""

    def test_acceptance_py_at_root_is_excluded(self, tmp_path):
        """Top-level acceptance.py with no __init__.py in its directory is NOT returned."""
        (tmp_path / "acceptance.py").write_text("import sys; sys.exit(0)\n")
        assert "acceptance.py" not in _names(_shipped_sources(tmp_path))

    def test_multiple_top_level_scripts_all_excluded(self, tmp_path):
        """Any number of top-level scripts are excluded when root has no __init__.py."""
        for name in ("acceptance.py", "main.py", "setup.py", "wsgi.py"):
            (tmp_path / name).write_text("pass\n")
        assert _shipped_sources(tmp_path) == []

    def test_top_level_script_alongside_package_is_still_excluded(self, tmp_path):
        """acceptance.py at root is excluded even when a package exists alongside it."""
        (tmp_path / "acceptance.py").write_text("x = 1\n")
        _write_package(tmp_path, "mypkg", "core.py")
        result = _shipped_sources(tmp_path)
        assert "acceptance.py" not in _names(result)
        assert "core.py" in _names(result)

    def test_root_with_init_py_includes_its_files(self, tmp_path):
        """Edge case: if root itself has __init__.py, files there ARE in scope."""
        (tmp_path / "__init__.py").write_text("")
        (tmp_path / "top.py").write_text("x = 1\n")
        assert "top.py" in _names(_shipped_sources(tmp_path))


# ---------------------------------------------------------------------------
# Core: package files included
# ---------------------------------------------------------------------------

class TestPackageFilesIncluded:
    """Files inside a directory containing __init__.py are always candidates."""

    def test_package_module_is_included(self, tmp_path):
        """A .py file inside a package (with __init__.py present) IS returned."""
        _write_package(tmp_path, "mypkg", "core.py")
        assert "core.py" in _names(_shipped_sources(tmp_path))

    def test_package_init_itself_is_included(self, tmp_path):
        """__init__.py inside a package is included (its parent contains it)."""
        _write_package(tmp_path, "mypkg")
        assert "__init__.py" in _names(_shipped_sources(tmp_path))

    def test_multiple_modules_in_same_package_all_included(self, tmp_path):
        """Every non-test module inside a package is included."""
        _write_package(tmp_path, "mypkg", "a.py", "b.py", "c.py")
        names = _names(_shipped_sources(tmp_path))
        assert "a.py" in names and "b.py" in names and "c.py" in names

    def test_nested_package_included(self, tmp_path):
        """Files in nested sub-packages are included when each level has __init__.py."""
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "__init__.py").write_text("")
        (inner / "__init__.py").write_text("")
        (inner / "deep.py").write_text("x = 42\n")
        assert "deep.py" in _names(_shipped_sources(tmp_path))

    def test_nested_file_excluded_when_immediate_parent_has_no_init(self, tmp_path):
        """File in nested dir is excluded if its IMMEDIATE parent has no __init__.py."""
        outer = tmp_path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        (outer / "__init__.py").write_text("")
        # inner/ intentionally lacks __init__.py
        (inner / "orphan.py").write_text("x = 1\n")
        assert "orphan.py" not in _names(_shipped_sources(tmp_path))


# ---------------------------------------------------------------------------
# Test files excluded from shipped sources
# ---------------------------------------------------------------------------

class TestTestFilesNeverShipped:
    """_is_test_file must keep all test artefacts out of the result."""

    def test_test_prefixed_file_excluded_inside_package(self, tmp_path):
        pkg = _write_package(tmp_path, "mypkg")
        (pkg / "test_something.py").write_text("def test_foo(): pass\n")
        assert "test_something.py" not in _names(_shipped_sources(tmp_path))

    def test_test_suffixed_file_excluded_inside_package(self, tmp_path):
        pkg = _write_package(tmp_path, "mypkg")
        (pkg / "something_test.py").write_text("def test_foo(): pass\n")
        assert "something_test.py" not in _names(_shipped_sources(tmp_path))

    def test_files_in_tests_dir_excluded(self, tmp_path):
        """Even if tests/ has __init__.py, its contents are test files."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "helper.py").write_text("def helper(): pass\n")
        assert "helper.py" not in _names(_shipped_sources(tmp_path))

    def test_files_in_test_dir_excluded(self, tmp_path):
        """Singular 'test/' directory also excluded."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "__init__.py").write_text("")
        (test_dir / "utils.py").write_text("x = 1\n")
        assert "utils.py" not in _names(_shipped_sources(tmp_path))


# ---------------------------------------------------------------------------
# Skip-dir filter
# ---------------------------------------------------------------------------

class TestSkipDirsFiltered:
    """Files inside _SKIP_DIRS must never appear regardless of __init__.py."""

    def test_pycache_excluded(self, tmp_path):
        cache = tmp_path / "__pycache__"
        cache.mkdir()
        (cache / "__init__.py").write_text("")
        (cache / "cached.py").write_text("x = 1\n")
        assert _shipped_sources(tmp_path) == []

    def test_venv_excluded(self, tmp_path):
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "__init__.py").write_text("")
        (venv / "lib.py").write_text("x = 1\n")
        assert _shipped_sources(tmp_path) == []

    def test_node_modules_excluded(self, tmp_path):
        nm = tmp_path / "node_modules"
        nm.mkdir()
        (nm / "__init__.py").write_text("")
        (nm / "script.py").write_text("x = 1\n")
        assert _shipped_sources(tmp_path) == []


# ---------------------------------------------------------------------------
# Result properties
# ---------------------------------------------------------------------------

class TestResultProperties:
    """Output list is sorted and contains only absolute paths."""

    def test_empty_project_returns_empty_list(self, tmp_path):
        assert _shipped_sources(tmp_path) == []

    def test_result_is_sorted(self, tmp_path):
        pkg = _write_package(tmp_path, "mypkg", "z_last.py", "a_first.py", "m_mid.py")
        result = _shipped_sources(tmp_path)
        assert result == sorted(result)

    def test_result_contains_path_objects(self, tmp_path):
        _write_package(tmp_path, "mypkg", "mod.py")
        for item in _shipped_sources(tmp_path):
            assert isinstance(item, Path)

    def test_result_paths_are_absolute(self, tmp_path):
        _write_package(tmp_path, "mypkg", "mod.py")
        for item in _shipped_sources(tmp_path):
            assert item.is_absolute()


# ---------------------------------------------------------------------------
# Integration: mutation gate with package-only mutable code
# ---------------------------------------------------------------------------

class TestMutationGatePackageOnlyProject:
    """
    Validates that the mutation gate passes for a project whose ONLY mutable
    source is inside a proper package, while a top-level acceptance.py (no
    __init__.py in its directory) is excluded from mutation entirely.
    """

    @pytest.fixture()
    def project(self, tmp_path) -> Path:
        """
        Minimal project layout:

          acceptance.py         ← top-level script, excluded by _shipped_sources
          conftest.py           ← adds sandbox root to sys.path for subprocess pytest
          mypkg/
            __init__.py
            calc.py             ← the only mutable source; has 4 mutation sites
          tests/
            __init__.py
            test_calc.py        ← kills all 4 mutants
        """
        # Top-level script — excluded from mutation by the package-only filter
        (tmp_path / "acceptance.py").write_text(textwrap.dedent("""\
            import sys
            sys.exit(0)
        """))

        # conftest.py so the subprocess pytest can import mypkg from the sandbox root
        (tmp_path / "conftest.py").write_text(textwrap.dedent("""\
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent))
        """))

        # Package with arithmetic and a comparison — 4 mutation sites total:
        #   site 0: BinOp  a+b   → a-b
        #   site 1: Return a+b   → None
        #   site 2: Compare x>0  → x<=0
        #   site 3: Return x>0   → None
        pkg = tmp_path / "mypkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "calc.py").write_text(textwrap.dedent("""\
            def add(a, b):
                return a + b

            def is_positive(x):
                return x > 0
        """))

        # Test suite that kills every one of the 4 mutants
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "__init__.py").write_text("")
        (tests / "test_calc.py").write_text(textwrap.dedent("""\
            import sys, os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from mypkg.calc import add, is_positive

            def test_add_gives_sum():
                assert add(2, 3) == 5          # kills a-b mutant (-1 ≠ 5)

            def test_add_not_none():
                assert add(10, 1) is not None  # kills return-None mutant
                assert add(10, 1) == 11

            def test_is_positive_true():
                assert is_positive(1) is True  # kills x<=0 and return-None mutants

            def test_is_positive_false():
                assert is_positive(-1) is False  # kills x<=0 mutant (True ≠ False)

            def test_is_positive_not_none():
                assert is_positive(5) is not None  # kills return-None mutant
        """))

        return tmp_path

    def test_acceptance_py_excluded_from_shipped_sources(self, project):
        """The top-level acceptance.py must NOT appear in _shipped_sources."""
        sources = _shipped_sources(project)
        assert "acceptance.py" not in _names(sources)

    def test_package_calc_included_in_shipped_sources(self, project):
        """calc.py inside mypkg/ must appear in _shipped_sources."""
        sources = _shipped_sources(project)
        assert "calc.py" in _names(sources)

    def test_mutation_gate_passes(self, project):
        """Gate must pass: the strong test suite kills all mutants in the package code."""
        result = run_mutation_gate(str(project), threshold=0.8, max_mutants=20)
        assert result.passed, f"Expected gate to pass but got: {result.detail}"

    def test_mutation_gate_name_is_mutation(self, project):
        result = run_mutation_gate(str(project), threshold=0.8, max_mutants=20)
        assert result.name == "mutation"

    def test_no_top_level_scripts_means_nothing_to_prove(self, tmp_path):
        """Project with ONLY top-level scripts passes trivially ('nothing to prove')."""
        (tmp_path / "acceptance.py").write_text("import sys; sys.exit(0)\n")
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_pass.py").write_text("def test_always(): pass\n")
        result = run_mutation_gate(str(tmp_path))
        assert result.passed
        assert "nothing to prove" in result.detail
