"""Tests for the report/ Typst document build.

Scope: the invariants that can silently drift between the document source and
the files it names. Compiling the document is deliberately out of scope -- it
needs the ``report`` pixi env (typst, snakemake), which the test env does not
have, and the build command is itself the verification. Build it by hand with
``pixi run report``.

The failure modes these exist to catch:

1. A figure reference in ``paper.typ`` that points at an asset PNG which is no
   longer there, or an asset PNG that nothing references. A file-per-asset
   layout invites exactly this, and in the second direction nothing else would
   ever notice: an orphaned PNG still gets committed, still counts toward the
   large-file limit, and never appears in the PDF.
2. A ``@key`` citation with no matching entry in ``biblio.bib``. Typst renders
   these as a visible marker rather than failing the build, so a broken
   citation ships unless something checks.
3. Non-ASCII bytes in the Typst sources. The repo-wide ASCII convention is
   enforced by ``scripts/check_ascii.py`` for ``*.py`` and ``*.md`` only, so
   ``.typ`` is not covered by the pre-commit hook. This is where that gap is
   closed.
4. Registry drift between the Snakefile's ``FIGURES`` and the
   ``figures/plot_*.py`` scripts on disk. Inert today because every figure is
   a committed asset and ``FIGURES`` is empty; written so it becomes
   load-bearing the moment the first generated figure lands.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / 'report'
ASSET_DIR = REPORT_DIR / 'assets'
FIGURE_DIR = REPORT_DIR / 'figures'
SNAKEFILE = REPORT_DIR / 'Snakefile'
BIBLIOGRAPHY = REPORT_DIR / 'biblio.bib'
PAPER = REPORT_DIR / 'paper.typ'

# report/*.typ plus report/lib/*.typ. Globbed rather than listed so a second
# document or a new lib module is covered without editing this file.
TYP_SOURCES = sorted(REPORT_DIR.glob('*.typ')) + sorted(
    (REPORT_DIR / 'lib').glob('*.typ')
)


def _paper_source() -> str:
    return PAPER.read_text(encoding='utf-8')


def _referenced_assets() -> set[str]:
    """Asset stems named by ``#asset("<name>", ...)`` calls in paper.typ."""
    return set(re.findall(r'#asset\(\s*"([^"]+)"', _paper_source()))


def _asset_stems_on_disk() -> set[str]:
    return {p.stem for p in ASSET_DIR.glob('*.png')}


def _cited_keys() -> set[str]:
    """Citation keys used in paper.typ.

    Typst spells a citation ``@key``. The same ``@`` sigil spells a
    cross-reference, so the label namespaces used for those (``eq:``, ``fig:``,
    ``sect:``, ``app:``) are excluded -- they are resolved by Typst against
    ``<label>`` definitions in the document, not against the bibliography.
    """
    keys = set(re.findall(r'(?<![\w@])@([A-Za-z][\w:]*)', _paper_source()))
    return {k for k in keys if ':' not in k}


def _bib_keys() -> set[str]:
    """Entry keys defined in biblio.bib."""
    text = BIBLIOGRAPHY.read_text(encoding='utf-8')
    return set(re.findall(r'@\w+\s*\{\s*([^,\s]+)\s*,', text))


def _snakefile_document_figures() -> dict[str, list[str]]:
    """Read the DOCUMENT_FIGURES mapping out of the Snakefile.

    Parsing the literal rather than importing keeps this test free of a
    snakemake dependency, which the test env does not have.
    """
    source = SNAKEFILE.read_text(encoding='utf-8')
    match = re.search(
        r'^DOCUMENT_FIGURES\s*=\s*(\{.*?\n\})', source, re.MULTILINE | re.DOTALL
    )
    if match is None:
        pytest.fail(f'No top-level `DOCUMENT_FIGURES = {{...}}` in {SNAKEFILE}')
    return ast.literal_eval(match.group(1))


def _registered_figures() -> set[str]:
    """The figure registry, derived exactly as the Snakefile derives it."""
    return {name for names in _snakefile_document_figures().values() for name in names}


def test_paper_source_exists():
    """The document exists, so the checks below are not vacuously true."""
    assert PAPER.is_file(), f'{PAPER} is missing'


def test_every_referenced_asset_exists():
    """Every PNG paper.typ names is present under report/assets/."""
    referenced = _referenced_assets()
    assert referenced, f'No #asset(...) calls found in {PAPER}'
    missing = sorted(referenced - _asset_stems_on_disk())
    assert not missing, (
        f'paper.typ references assets with no matching .png in {ASSET_DIR}: {missing}'
    )


def test_every_asset_is_referenced():
    """Every PNG under report/assets/ is named by paper.typ.

    The other direction of the same drift. An unreferenced asset is a
    committed binary that never reaches the PDF, and nothing else would ever
    report it.
    """
    on_disk = _asset_stems_on_disk()
    assert on_disk, f'No .png assets found in {ASSET_DIR}'
    unreferenced = sorted(on_disk - _referenced_assets())
    assert not unreferenced, (
        f'Assets in {ASSET_DIR} that paper.typ never references: {unreferenced}. '
        f'Reference them, or delete them.'
    )


def test_every_citation_resolves():
    """Every @key citation in paper.typ has an entry in biblio.bib.

    Typst renders an unresolved citation as a visible marker instead of
    failing the build, so this is the only thing that catches a typo'd key
    before it reaches a reader.
    """
    cited = _cited_keys()
    assert cited, f'No citations found in {PAPER}'
    unresolved = sorted(cited - _bib_keys())
    assert not unresolved, (
        f'Citations in paper.typ with no entry in {BIBLIOGRAPHY}: {unresolved}'
    )


@pytest.mark.parametrize('source', TYP_SOURCES, ids=lambda p: p.name)
def test_typ_sources_are_ascii(source: Path):
    """No non-ASCII bytes in the Typst sources.

    scripts/check_ascii.py covers *.py and *.md only, so .typ files are not
    reached by the pre-commit hook. Math notation belongs in Typst markup
    (``arrow(q)``, ``<=``, ``plus.minus``), not as Unicode glyphs.
    """
    violations = [
        f'line {lineno}, col {col}: {ch!r} (U+{ord(ch):04X})'
        for lineno, line in enumerate(
            source.read_text(encoding='utf-8').splitlines(), start=1
        )
        for col, ch in enumerate(line, start=1)
        if ord(ch) > 127
    ]
    assert not violations, f'Non-ASCII characters in {source}:\n' + '\n'.join(
        violations
    )


def test_documents_in_snakefile_exist():
    """Every document the Snakefile builds has a .typ source on disk."""
    for document in _snakefile_document_figures():
        source = REPORT_DIR / f'{document}.typ'
        assert source.is_file(), f'DOCUMENT_FIGURES lists missing {source}'


def test_every_registered_figure_has_a_script():
    """Every name in the Snakefile's FIGURES maps to a script on disk.

    Vacuously true while FIGURES is empty -- every figure is a committed asset
    today. It becomes load-bearing as soon as a generated figure is
    registered, which is the point of writing it now.
    """
    missing = sorted(
        name
        for name in _registered_figures()
        if not (FIGURE_DIR / f'plot_{name}.py').is_file()
    )
    assert not missing, (
        f'Snakefile FIGURES names with no plot_<name>.py in {FIGURE_DIR}: {missing}'
    )


def test_every_script_is_registered():
    """Every plot_*.py on disk appears in the Snakefile's FIGURES.

    The other direction: an unregistered script is dead weight that never gets
    built and never gets noticed.
    """
    on_disk = {p.stem.removeprefix('plot_') for p in FIGURE_DIR.glob('plot_*.py')}
    unregistered = sorted(on_disk - _registered_figures())
    assert not unregistered, (
        f'plot_*.py scripts absent from the Snakefile FIGURES list: {unregistered}. '
        f'Add them to a document in DOCUMENT_FIGURES, or delete them.'
    )
