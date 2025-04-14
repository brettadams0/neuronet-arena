# main.py
# 🧠 Entry point for NeuroNet Arena

from engine.simulator import Simulation


def main():
    # Configuration
    NUM_AGENTS = 20
    NUM_NODES = 50
    GENERATIONS = 100
    STEPS_PER_GENERATION = 300
    ARENA_WIDTH = 800
    ARENA_HEIGHT = 600

    # Initialize and run the simulation
    sim = Simulation(
        num_agents=NUM_AGENTS,
        num_nodes=NUM_NODES,
        generations=GENERATIONS,
        steps_per_generation=STEPS_PER_GENERATION,
        arena_width=ARENA_WIDTH,
        arena_height=ARENA_HEIGHT
    )
    sim.run()


if __name__ == '__main__':
    main()
