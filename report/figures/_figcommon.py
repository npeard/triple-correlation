"""Shared setup for the report's generated figures.

Every `plot_<name>.py` in this directory is a standalone script that the
Snakemake `figure` rule invokes as `python plot_<name>.py --output <path>`
and that writes exactly the path it is handed. This module holds the three
things all of them need: the headless backend, the argument parsing, and the
save.

The backend selection is the reason this module must be imported before
anything pulls in `matplotlib.pyplot`. The DAG runs with no display, so
importing pyplot first would bind an interactive backend and fail.
"""

import argparse
import warnings
from pathlib import Path

import matplotlib

# Must precede any pyplot import, including the transitive one through
# plots.PaperFigures. Agg is the only backend guaranteed present and
# display-free on both win-64 and linux-64.
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Cheap guard on the import order above. If a future edit hoists the pyplot
# import over `matplotlib.use`, the backend silently reverts to an interactive
# one and every figure body fails on a machine with no display -- a failure
# that would surface in CI, not here. Asserting it fails loudly at the source
# instead.
if plt.get_backend().lower() != 'agg':
    msg = (
        f'Expected the Agg backend, got {plt.get_backend()!r}. Something '
        f'imported matplotlib.pyplot before this module called '
        f'matplotlib.use("Agg").'
    )
    raise RuntimeError(msg)

# The published figures were rendered at this DPI. Kept explicit so a
# matplotlib default change cannot silently rescale every figure in the paper.
DPI = 200

# The figure bodies in plots/PaperFigures.py all end in `P.show()`, which under
# Agg warns once per call and does nothing. These scripts save the figure
# themselves, so the call is dead weight. Filtered on the exact message rather
# than by category, so every other UserWarning still surfaces -- a clean
# report build should emit no warnings, which only stays a useful signal if
# real ones are not swallowed alongside this one.
warnings.filterwarnings(
    'ignore',
    message='FigureCanvasAgg is non-interactive, and thus cannot be shown',
    category=UserWarning,
)


def output_path() -> Path:
    """Parses `--output` and returns the single path this script must write.

    Returns:
        The path passed on the command line, with its parent created.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='PDF path to write')
    path = parser.parse_args().output
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save(path: Path, figure: plt.Figure | None = None) -> None:
    """Writes one figure to `path` and closes it.

    Args:
        path: Destination, normally from `output_path()`.
        figure: Figure to write; the current figure when omitted.

    Raises:
        RuntimeError: If the body left no figure open, or left more than one
            without the caller naming which to keep.
    """
    open_figures = plt.get_fignums()
    if figure is None:
        # A body that opens several figures is a composite that was assembled
        # outside this repo (Figure_4_Rows and Figure_S1 both do this). Saving
        # `gcf()` there silently writes one panel of several and the PDF looks
        # plausible, so refuse instead of guessing.
        if not open_figures:
            msg = f'Figure body opened no figure, so nothing can be written to {path}.'
            raise RuntimeError(msg)
        if len(open_figures) > 1:
            msg = (
                f'Figure body left {len(open_figures)} figures open; saving the '
                f'current one would write a single panel of a composite. Pass '
                f'the intended figure explicitly.'
            )
            raise RuntimeError(msg)
        figure = plt.gcf()

    # CreationDate is the only byte that varies between two runs of the same
    # script, and Snakemake 9 hashes input *content*: left in, every figure
    # looks changed on every run and the DAG's caching is worthless. Setting it
    # to None omits the key entirely.
    figure.savefig(path, dpi=DPI, bbox_inches='tight', metadata={'CreationDate': None})
    plt.close('all')
