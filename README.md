# neuronet-arena

An evolution simulator. Twenty agents move around a 2D arena scattered with food and hazards; each
one is steered by its own small feedforward network. At the end of a generation the highest-scoring
agents are cloned, their weights are perturbed, and the next generation runs. Rendered live in
Pygame.

The point of the project is that nothing here is a library call — the network, the forward pass, and
the mutation operator are all written out in plain Python over nested lists. No PyTorch, no NumPy.
It is much slower than a framework would be, and much clearer about what a "brain" actually is.

<img src="img/gameplay.png" width="700"/>

## Running it

```sh
pip install -r requirements.txt   # pygame
python main.py
```

Configuration lives at the top of `main.py`: 20 agents, 50 nodes, 100 generations, 300 steps per
generation, in an 800x600 arena.

## The network

`engine/brain.py` — 6 inputs → 12 hidden (tanh) → 2 outputs (sigmoid), Xavier-initialised.

Each tick, `Agent.sense` builds the input vector:

| Input | |
|---|---|
| 0 | Food nodes within vision radius, scaled by 1/10 |
| 1 | Hazard nodes within vision radius, scaled by 1/10 |
| 2–3 | Mean offset to visible nodes, x and y, scaled by 1/100 |
| 4–5 | Current heading as `cos(θ)`, `sin(θ)` |

`decide` maps the first output to a turn of at most ±π/16 and applies it; `act` steps forward along
the new heading and resolves any node the agent lands on. Fitness accumulates from those node
interactions.

Worth noting: the network declares two outputs but `decide` only reads `output[0]`. The second is
computed and thrown away — a spare lever for something like a speed control, currently unused.

## Evolution

`engine/evolution.py` keeps the top performers, clones their brains, and calls `Brain.mutate`, which
walks every weight and bias and adds uniform noise in `[-strength, +strength]` with probability
`rate` (defaults `0.1` / `0.5`). Straight hill-climbing with cloning — no crossover between parents.

## Layout

```
main.py                 entry point and configuration
engine/simulator.py     generation loop
engine/arena.py         world generation, node placement
engine/agent.py         sense / decide / act
engine/brain.py         the network
engine/evolution.py     selection and mutation
engine/visualizer.py    Pygame rendering
```

## Where it could go

Crossover between two parents rather than clone-and-mutate. Memory, so an agent's state carries
across ticks. A fitness-over-time plot, which would show whether evolution is actually working or
just drifting.

## License

MIT — see [LICENSE](LICENSE).
