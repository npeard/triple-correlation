"""figure_1 / fig:Intro -- coherent and incoherent diffraction schematic.

Renders `plots.PaperFigures.Figure_1`, which draws the published schematic of
forward-direction coherent diffraction alongside incoherent frames collected
from the side.

The figure body is called rather than copied: it is the published render's
source, and duplicating it here would leave two definitions of one figure to
keep in step.
"""

import _figcommon

from plots import PaperFigures

if __name__ == '__main__':
    path = _figcommon.output_path()
    PaperFigures.Figure_1()
    _figcommon.save(path)
