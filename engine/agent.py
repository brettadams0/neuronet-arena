# engine/agent.py
# 🤖 Agent logic: AI-controlled entity with sensors, decision-making, and action

import random
import math
from engine.brain import Brain

class Agent:
    def __init__(self, position, brain=None):
        self.x, self.y = position
        self.fitness = 0
        self.vision_radius = 75
        self.speed = 3
        self.brain = brain if brain else Brain(input_size=6, hidden_size=12, output_size=2)
        self.direction = random.uniform(0, 2 * math.pi)  # radians

        self.sensed_data = [0.0] * 6

    def sense(self, arena, agents):
        visible_nodes = arena.get_nearby_nodes(self.x, self.y, self.vision_radius)
        food_count = sum(1 for node in visible_nodes if node.node_type == 'food')
        hazard_count = sum(1 for node in visible_nodes if node.node_type == 'hazard')

        avg_dx = sum(node.x - self.x for node in visible_nodes) / (len(visible_nodes) + 1)
        avg_dy = sum(node.y - self.y for node in visible_nodes) / (len(visible_nodes) + 1)

        self.sensed_data = [
            food_count / 10.0,
            hazard_count / 10.0,
            avg_dx / 100.0,
            avg_dy / 100.0,
            math.cos(self.direction),
            math.sin(self.direction),
        ]

    def decide(self):
        output = self.brain.forward(self.sensed_data)
        turn = (output[0] - 0.5) * 2 * math.pi / 16  # [-pi/16, pi/16]
        self.direction += turn

    def act(self, arena):
        self.x += math.cos(self.direction) * self.speed
        self.y += math.sin(self.direction) * self.speed
        self.x = max(0, min(arena.width, self.x))
        self.y = max(0, min(arena.height, self.y))

        nearby = arena.get_nearby_nodes(self.x, self.y, 10)
        for node in nearby:
            arena.resolve_node_effect(node.node_type, self)

    @property
    def position(self):
        return (int(self.x), int(self.y))
