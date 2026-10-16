"""Evolution simulator: small neural networks, a genetic algorithm, no ML framework."""

from neuronet.config import Config
from neuronet.rng import Rng
from neuronet.simulation import GenerationStats, Simulation

__all__ = ["Config", "GenerationStats", "Rng", "Simulation"]
__version__ = "2.0.0"
