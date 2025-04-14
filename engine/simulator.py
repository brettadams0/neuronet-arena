# engine/simulator.py
# 🎮 Simulation controller: handles generation loop, agent updates, evolution, and rendering

import pygame
from engine.arena import Arena
from engine.agent import Agent
from engine.evolution import evolve_population
from engine.visualizer import Visualizer

class Simulation:
    def __init__(self, num_agents, num_nodes, generations, steps_per_generation, arena_width, arena_height):
        self.num_agents = num_agents
        self.num_nodes = num_nodes
        self.generations = generations
        self.steps_per_generation = steps_per_generation
        self.arena_width = arena_width
        self.arena_height = arena_height

        self.agents = []
        self.arena = None
        self.visualizer = None

    def setup(self):
        self.arena = Arena(self.num_nodes, self.arena_width, self.arena_height)
        self.agents = [Agent(self.arena.random_spawn_point()) for _ in range(self.num_agents)]
        self.visualizer = Visualizer(self.arena, self.agents, self.arena_width, self.arena_height)

    def run(self):
        pygame.init()
        clock = pygame.time.Clock()

        for gen in range(self.generations):
            print(f"\n🔁 Generation {gen + 1}/{self.generations}")
            self.setup()
            for step in range(self.steps_per_generation):
                self.update_agents()
                self.visualizer.draw(step, gen)
                clock.tick(30)  # Cap at 30 FPS

            self.evolve_agents()

        pygame.quit()

    def update_agents(self):
        for agent in self.agents:
            agent.sense(self.arena, self.agents)
            agent.decide()
            agent.act(self.arena)

    def evolve_agents(self):
        print("\n🧬 Evolving agents...")
        brains = [agent.brain for agent in self.agents]
        fitnesses = [agent.fitness for agent in self.agents]
        new_brains = evolve_population(brains, fitnesses)

        # Respawn agents with new brains
        self.agents = [Agent(self.arena.random_spawn_point(), brain=new_brains[i])
                       for i in range(self.num_agents)]
