"""figure_11 / fig:PhaseTogglingAppendix -- diagonal resolving.

Renders `plots.PaperFigures.Figure_ResolvingDemo`, the source of the published render.

The figure body is called rather than copied: duplicating it here would leave
two definitions of one figure to keep in step.
"""

import _figcommon

from plots import PaperFigures

if __name__ == '__main__':
    path = _figcommon.output_path()
    PaperFigures.Figure_ResolvingDemo()
    _figcommon.save(path)
