"""Watch a run live in a pygame window.

Controls: space pauses, up/down change how many simulation steps are drawn
per frame, F finishes the current generation instantly, Esc or Q quits.

This module only reads the simulation; everything it shows comes from the
same headless engine that ``neuronet train`` runs.
"""

from __future__ import annotations

import pygame

from neuronet.brain import INPUT_LABELS, OUTPUT_LABELS
from neuronet.evolution import rank
from neuronet.simulation import Simulation

PANEL = 300
BG = (11, 15, 20)
SURFACE = (19, 26, 34)
GRID = (28, 36, 46)
TEXT = (231, 236, 242)
MUTED = (134, 149, 167)
AMBER = (255, 180, 84)
TEAL = (79, 214, 196)
FOOD = (79, 214, 196)
HAZARD = (232, 96, 86)
AGENT = (190, 200, 212)
SPEEDS = (1, 2, 4, 8, 16, 32, 64)


def run_viewer(sim: Simulation) -> None:
    cfg = sim.cfg
    pygame.init()
    pygame.display.set_caption("neuronet-arena")
    width, height = int(cfg.width), int(cfg.height)
    screen = pygame.display.set_mode((width + PANEL, height))
    font = pygame.font.SysFont("ibmplexmono,dejavusansmono,menlo,consolas,monospace", 14)
    big = pygame.font.SysFont("ibmplexsans,dejavusans,helvetica,arial", 22, bold=True)
    clock = pygame.time.Clock()

    speed_index = 0
    paused = False
    running = True
    sim.begin_generation()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    speed_index = min(speed_index + 1, len(SPEEDS) - 1)
                elif event.key == pygame.K_DOWN:
                    speed_index = max(speed_index - 1, 0)
                elif event.key == pygame.K_f:
                    while not sim.done:
                        sim.step()

        if not paused:
            for _ in range(SPEEDS[speed_index]):
                if sim.done:
                    sim.end_generation()
                    sim.begin_generation()
                    break
                sim.step()

        _draw_arena(screen, sim)
        _draw_panel(screen, sim, font, big, SPEEDS[speed_index], paused)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def _draw_arena(screen: pygame.Surface, sim: Simulation) -> None:
    cfg = sim.cfg
    world = sim.world
    screen.fill(BG)
    for gx in range(0, int(cfg.width), 60):
        pygame.draw.line(screen, GRID, (gx, 0), (gx, int(cfg.height)))
    for gy in range(0, int(cfg.height), 60):
        pygame.draw.line(screen, GRID, (0, gy), (int(cfg.width), gy))
    if world is None:
        return

    for x, y in zip(world.hazard_x, world.hazard_y, strict=True):
        pygame.draw.circle(screen, (60, 26, 26), (x, y), cfg.hazard_radius + 4)
        pygame.draw.circle(screen, HAZARD, (x, y), cfg.hazard_radius)
    for x, y in zip(world.food_x, world.food_y, strict=True):
        pygame.draw.circle(screen, FOOD, (x, y), cfg.food_radius)

    leader = rank([a.fitness for a in sim.agents])[0] if sim.agents else -1
    for i, a in enumerate(sim.agents):
        if not a.alive:
            pygame.draw.circle(screen, (60, 66, 76), (a.x, a.y), 3)
            continue
        colour = AMBER if i == leader else AGENT
        r = cfg.agent_radius
        tip = (a.x + a.hx * r * 1.8, a.y + a.hy * r * 1.8)
        left = (a.x - a.hx * r + a.hy * r, a.y - a.hy * r - a.hx * r)
        right = (a.x - a.hx * r - a.hy * r, a.y - a.hy * r + a.hx * r)
        pygame.draw.polygon(screen, colour, (tip, left, right))
        if i == leader:
            pygame.draw.circle(screen, (70, 56, 30), (a.x, a.y), cfg.vision, 1)


def _draw_panel(
    screen: pygame.Surface,
    sim: Simulation,
    font: pygame.font.Font,
    big: pygame.font.Font,
    speed: int,
    paused: bool,
) -> None:
    cfg = sim.cfg
    x0 = int(cfg.width)
    pygame.draw.rect(screen, SURFACE, (x0, 0, PANEL, int(cfg.height)))

    def text(s: str, x: int, y: int, colour: tuple[int, int, int] = TEXT) -> None:
        screen.blit(font.render(s, True, colour), (x0 + x, y))

    screen.blit(big.render(f"Generation {sim.generation}", True, TEXT), (x0 + 16, 14))
    alive = sum(1 for a in sim.agents if a.alive)
    text(f"step {sim.step_index}/{cfg.steps}   alive {alive}/{len(sim.agents)}", 16, 46, MUTED)
    text(f"{'paused' if paused else f'{speed}x'}   space f up/down q", 16, 64, MUTED)

    # Food per agent by generation: mean in amber, best in teal.
    top, h, w = 100, 120, PANEL - 32
    pygame.draw.rect(screen, BG, (x0 + 16, top, w, h))
    hist = sim.history[-w:]
    if hist:
        peak = max(max(s.best for s in hist), 1)
        for field, colour in (("best", TEAL), ("mean", AMBER)):
            pts = [
                (x0 + 16 + i * w / max(1, len(hist) - 1), top + h - getattr(s, field) / peak * h)
                for i, s in enumerate(hist)
            ]
            if len(pts) > 1:
                pygame.draw.lines(screen, colour, False, pts, 2)
        last = hist[-1]
        text(f"mean {last.mean:5.2f}", 16, top + h + 6, AMBER)
        text(f"best {last.best:3d}", 140, top + h + 6, TEAL)
    else:
        text("food per agent appears here", 24, top + h // 2 - 8, MUTED)

    # The current leader's inputs and outputs.
    if not sim.agents:
        return
    leader = sim.agents[rank([a.fitness for a in sim.agents])[0]]
    y = top + h + 40
    text(f"leader: {leader.food_eaten} food, energy {leader.energy:5.1f}", 16, y)
    y += 24
    values = [*leader.inputs, leader.steer, leader.throttle]
    labels = [*INPUT_LABELS, *OUTPUT_LABELS]
    for label, value in zip(labels, values, strict=True):
        text(f"{label:<14}", 16, y, MUTED)
        bar = int(max(-1.0, min(1.0, value)) * 60)
        colour = AMBER if value >= 0 else TEAL
        mid = x0 + 200
        pygame.draw.rect(screen, colour, (mid + min(0, bar), y + 3, abs(bar), 10))
        pygame.draw.line(screen, MUTED, (mid, y + 1), (mid, y + 14))
        y += 20
