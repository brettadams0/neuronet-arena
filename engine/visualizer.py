# engine/visualizer.py
# 🖼️ Pygame-based live visual rendering of the simulation

import pygame

class Visualizer:
    def __init__(self, arena, agents, width, height):
        self.arena = arena
        self.agents = agents
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("NeuroNet Arena")
        self.bg_color = (10, 10, 30)
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, step, generation):
        self.screen.fill(self.bg_color)
        self._draw_nodes()
        self._draw_agents()
        self._draw_hud(step, generation)
        pygame.display.flip()

    def _draw_nodes(self):
        for node in self.arena.get_nodes():
            color = (200, 200, 200)
            if node.node_type == 'food':
                color = (50, 220, 50)
            elif node.node_type == 'hazard':
                color = (220, 50, 50)
            pygame.draw.circle(self.screen, color, (node.x, node.y), 5)

    def _draw_agents(self):
        for agent in self.agents:
            x, y = agent.position
            pygame.draw.circle(self.screen, (100, 200, 255), (x, y), 6)
            end_x = int(x + 10 * pygame.math.Vector2(1, 0).rotate_rad(agent.direction).x)
            end_y = int(y + 10 * pygame.math.Vector2(1, 0).rotate_rad(agent.direction).y)
            pygame.draw.line(self.screen, (255, 255, 255), (x, y), (end_x, end_y), 2)

    def _draw_hud(self, step, generation):
        text = self.font.render(f"Gen {generation+1} | Step {step+1}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
