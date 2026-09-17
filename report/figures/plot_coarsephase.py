"""figure_8 / fig:CoarsePhase -- phase at 201x201 against 29x29.

Renders `plots.PaperFigures.Figure_S4`, the source of the published render.

The figure body is called rather than copied: duplicating it here would leave
two definitions of one figure to keep in step.
"""

import _figcommon

from plots import PaperFigures

if __name__ == '__main__':
    path = _figcommon.output_path()
    PaperFigures.Figure_S4()
    _figcommon.save(path)
