# engine/arena.py
# 🌐 Arena logic: graph-based world with resource and hazard nodes

import random
import math

class Node:
    def __init__(self, x, y, node_type='neutral'):
        self.x = x
        self.y = y
        self.node_type = node_type  # 'food', 'hazard', or 'neutral'

    def position(self):
        return (self.x, self.y)

class Arena:
    def __init__(self, num_nodes, width, height):
        self.width = width
        self.height = height
        self.nodes = []
        self._generate_nodes(num_nodes)

    def _generate_nodes(self, num_nodes):
        for _ in range(num_nodes):
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            node_type = random.choices(
                ['neutral', 'food', 'hazard'],
                weights=[0.6, 0.3, 0.1],
                k=1
            )[0]
            self.nodes.append(Node(x, y, node_type))

    def random_spawn_point(self):
        return random.choice(self.nodes).position()

    def get_nearby_nodes(self, x, y, radius):
        return [node for node in self.nodes if self._distance(node.x, node.y, x, y) < radius]

    def _distance(self, x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def resolve_node_effect(self, node_type, agent):
        if node_type == 'food':
            agent.fitness += 1
        elif node_type == 'hazard':
            agent.fitness -= 1

    def get_nodes(self):
        return self.nodes
