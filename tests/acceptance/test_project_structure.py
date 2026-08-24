"""Tests: verify tts_studio deletion and acceptance file structure."""
from __future__ import annotations

import re
from pathlib import Path


_PROJECT_ROOT = Path(__file__).parents[2]

# Patterns that indicate test-only stubs rather than real implementations.
_FAKE_PATTERN = re.compile(
    r"\b(mock|fake|dummy|stub)\b",
    re.IGNORECASE,
)


def _content_lines(path: Path) -> list[str]:
    """Return non-blank, non-comment lines from a Python source file."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return lines


# ---------------------------------------------------------------------------
# tts_studio must not exist
# ---------------------------------------------------------------------------

class TestNoTtsStudioDirectory:
    def test_tts_studio_absent_from_project_root(self):
        """tts_studio/ must not be present; it was removed as part of cleanup."""
        candidate = _PROJECT_ROOT / "tts_studio"
        assert not candidate.exists(), (
            f"Found {candidate} — the tts_studio/ directory must not exist under the project root"
        )

    def test_tts_studio_is_not_a_file_either(self):
        """Ensure there is no file named tts_studio at the project root."""
        candidate = _PROJECT_ROOT / "tts_studio"
        assert not candidate.is_file(), (
            "tts_studio exists as a file at the project root; it must be absent entirely"
        )

    def test_no_tts_studio_subtree_anywhere_under_root(self):
        """There should be no tts_studio directory anywhere inside the project tree."""
        matches = list(_PROJECT_ROOT.rglob("tts_studio"))
        assert matches == [], (
            f"tts_studio found inside project tree: {matches}"
        )


class TestCanonicalRepositoryLayout:
    def test_application_packages_are_explicit(self):
        assert (
            _PROJECT_ROOT / "dubbing" / "apps" / "personal_capture"
        ).is_dir()
        assert (_PROJECT_ROOT / "dubbing" / "apps" / "dubbing_web.py").is_file()
        assert not (_PROJECT_ROOT / "dubbing" / "capture").exists()
        assert not (_PROJECT_ROOT / "dubbing" / "web.py").exists()

    def test_cli_has_its_own_package(self):
        assert (_PROJECT_ROOT / "dubbing" / "cli" / "main.py").is_file()
        assert (_PROJECT_ROOT / "dubbing" / "__main__.py").is_file()

    def test_product_documents_are_not_loose_at_root(self):
        for legacy in (
            "DESIGN.md",
            "PRODUCT.md",
            "TRANSCRIPTION_IMPLEMENTATION_REPORT.md",
            "sample.srt",
        ):
            assert not (_PROJECT_ROOT / legacy).exists()
        assert (_PROJECT_ROOT / "docs" / "ARCHITECTURE.md").is_file()
        assert (_PROJECT_ROOT / "docs" / "CERTIFICATION.md").is_file()
        assert (_PROJECT_ROOT / "samples" / "sample.srt").is_file()

    def test_tests_mirror_product_boundaries(self):
        for group in (
            "acceptance",
            "cli",
            "core",
            "diarization",
            "personal_capture",
            "transcription",
            "translation",
            "web",
        ):
            assert (_PROJECT_ROOT / "tests" / group).is_dir()

    def test_deployment_assets_have_product_and_host_ownership(self):
        deployment = (
            _PROJECT_ROOT / "deploy" / "personal_capture" / "gigabyte"
        )
        assert (deployment / "personal-capture.json").is_file()
        assert (deployment / "dubbing-capture-watch.service").is_file()
        assert (deployment / "dubbing-capture-review.service").is_file()


# ---------------------------------------------------------------------------
# Acceptance file must exist
# ---------------------------------------------------------------------------

class TestAcceptanceFileExists:
    def test_acceptance_py_exists(self):
        """acceptance.py must be present at the project root."""
        assert (_PROJECT_ROOT / "acceptance.py").is_file(), (
            "acceptance.py does not exist at the project root"
        )

    def test_acceptance_script_pointer_exists(self):
        """The 'acceptance' pointer file that names acceptance.py must exist."""
        assert (_PROJECT_ROOT / "acceptance").exists(), (
            "'acceptance' pointer file is missing from the project root"
        )

    def test_acceptance_pointer_references_acceptance_py(self):
        """The 'acceptance' file must reference acceptance.py so the runner knows what to call."""
        content = (_PROJECT_ROOT / "acceptance").read_text(encoding="utf-8")
        assert "acceptance.py" in content, (
            f"'acceptance' file does not mention acceptance.py; got: {content!r}"
        )

    def test_acceptance_py_is_not_empty(self):
        """acceptance.py must contain actual code, not an empty placeholder."""
        content = (_PROJECT_ROOT / "acceptance.py").read_text(encoding="utf-8").strip()
        assert content, "acceptance.py is empty"


# ---------------------------------------------------------------------------
# Acceptance file must contain no mock / fake / dummy / stub patterns
# ---------------------------------------------------------------------------

class TestAcceptanceFileClean:
    def test_no_mock_pattern_in_acceptance_py(self):
        """Non-comment, non-blank lines in acceptance.py must not contain 'mock'."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        offending = [ln for ln in lines if re.search(r"\bmock\b", ln, re.IGNORECASE)]
        assert not offending, (
            f"acceptance.py contains 'mock' on non-comment lines: {offending}"
        )

    def test_no_fake_pattern_in_acceptance_py(self):
        """Non-comment, non-blank lines in acceptance.py must not contain 'fake'."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        offending = [ln for ln in lines if re.search(r"\bfake\b", ln, re.IGNORECASE)]
        assert not offending, (
            f"acceptance.py contains 'fake' on non-comment lines: {offending}"
        )

    def test_no_dummy_pattern_in_acceptance_py(self):
        """Non-comment, non-blank lines in acceptance.py must not contain 'dummy'."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        offending = [ln for ln in lines if re.search(r"\bdummy\b", ln, re.IGNORECASE)]
        assert not offending, (
            f"acceptance.py contains 'dummy' on non-comment lines: {offending}"
        )

    def test_no_stub_pattern_in_acceptance_py(self):
        """Non-comment, non-blank lines in acceptance.py must not contain 'stub'."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        offending = [ln for ln in lines if re.search(r"\bstub\b", ln, re.IGNORECASE)]
        assert not offending, (
            f"acceptance.py contains 'stub' on non-comment lines: {offending}"
        )

    def test_acceptance_py_combined_fake_patterns_absent(self):
        """Single sweep: none of mock/fake/dummy/stub appear on any content line."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        offending = [(i, ln) for i, ln in enumerate(lines, 1) if _FAKE_PATTERN.search(ln)]
        assert not offending, (
            "acceptance.py has fake/stub/mock/dummy patterns on content lines: "
            + "; ".join(f"line-approx {i}: {ln!r}" for i, ln in offending)
        )

    def test_content_lines_helper_skips_blank_lines(self):
        """Verify _content_lines properly ignores blank lines so the scan is accurate."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        assert all(ln.strip() for ln in lines), "content lines list contains blank entries"

    def test_content_lines_helper_skips_comment_lines(self):
        """Verify _content_lines properly ignores # comment lines."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        assert all(not ln.startswith("#") for ln in lines), (
            "content lines list contains # comment entries"
        )

    def test_acceptance_py_has_real_code(self):
        """acceptance.py must have at least some non-comment, non-blank content lines."""
        lines = _content_lines(_PROJECT_ROOT / "acceptance.py")
        assert len(lines) >= 5, (
            f"acceptance.py has too few content lines ({len(lines)}); expected real implementation"
        )

    def test_acceptance_py_uses_real_subprocess_or_import(self):
        """acceptance.py must import subprocess or run real commands — not just print."""
        content = (_PROJECT_ROOT / "acceptance.py").read_text(encoding="utf-8")
        assert "subprocess" in content or "urllib" in content, (
            "acceptance.py does not appear to run real system checks "
            "(no 'subprocess' or 'urllib' found)"
        )
