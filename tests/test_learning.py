"""The claim the project rests on: selection, not luck, is what improves foraging.

Slow (about half a minute), so it carries the ``slow`` marker; CI runs it.
"""

import pytest

from neuronet.config import Config
from neuronet.simulation import Simulation

GENERATIONS = 30


@pytest.mark.slow
def test_selection_beats_the_no_selection_control() -> None:
    def late_mean(selection: str) -> float:
        history = Simulation(Config(selection=selection), seed=3).run(GENERATIONS)  # type: ignore[arg-type]
        late = history[-5:]
        return sum(s.mean for s in late) / len(late)

    evolved, control = late_mean("tournament"), late_mean("random")
    assert evolved > 3 * control, (evolved, control)
