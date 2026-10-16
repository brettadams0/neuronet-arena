import math
from collections.abc import Sequence

import pytest

from neuronet.agent import Agent
from neuronet.brain import Brain, genome_size, layer_sizes
from neuronet.config import Config
from neuronet.rng import Rng
from neuronet.world import World

CFG = Config(food_count=0, hazard_count=0)


def _world(
    food: Sequence[tuple[float, float]] = (), hazards: Sequence[tuple[float, float]] = ()
) -> World:
    world = World(CFG, Rng(0))
    world.food_x = [p[0] for p in food]
    world.food_y = [p[1] for p in food]
    world.hazard_x = [p[0] for p in hazards]
    world.hazard_y = [p[1] for p in hazards]
    return world


def _agent(x: float = 480.0, y: float = 300.0, hx: float = 1.0, hy: float = 0.0) -> Agent:
    layers = layer_sizes(CFG.hidden)
    return Agent(Brain(layers, [0.0] * genome_size(layers)), x, y, hx, hy, CFG.start_energy)


# -- senses ----------------------------------------------------------------


def test_nothing_in_sight_reads_zero() -> None:
    inputs = _agent().sense(_world())
    assert inputs[:6] == [0.0] * 6


def test_food_dead_ahead() -> None:
    near, ahead, right = _agent().sense(_world(food=[(550.0, 300.0)]))[:3]
    assert near == pytest.approx(1 - 70 / CFG.vision)
    assert (ahead, right) == (1.0, 0.0)


def test_food_to_the_right_in_screen_coordinates() -> None:
    # Facing +x with y pointing down the screen, +y is on the agent's right.
    _, ahead, right = _agent().sense(_world(food=[(480.0, 350.0)]))[:3]
    assert ahead == pytest.approx(0.0)
    assert right == pytest.approx(1.0)


def test_senses_rotate_with_the_agent() -> None:
    # Same food, agent now facing +y (down): the food is dead ahead.
    _, ahead, right = _agent(hx=0.0, hy=1.0).sense(_world(food=[(480.0, 350.0)]))[:3]
    assert (ahead, right) == pytest.approx((1.0, 0.0))


def test_only_the_nearest_item_is_reported() -> None:
    near = _agent().sense(_world(food=[(600.0, 300.0), (520.0, 300.0)]))[0]
    assert near == pytest.approx(1 - 40 / CFG.vision)


def test_items_beyond_vision_are_invisible() -> None:
    assert _agent().sense(_world(food=[(480.0 + CFG.vision + 1, 300.0)]))[:3] == [0.0] * 3


def test_hazards_have_their_own_channels() -> None:
    inputs = _agent().sense(_world(hazards=[(480.0, 230.0)]))
    assert inputs[:3] == [0.0] * 3
    assert inputs[3] == pytest.approx(1 - 70 / CFG.vision)
    assert inputs[5] == pytest.approx(-1.0)  # above the agent is on its left


def test_wall_ahead() -> None:
    agent = _agent(x=CFG.width - 40.0)
    assert agent.sense(_world())[6] == pytest.approx(1 - 40 / CFG.vision)
    agent = _agent(x=CFG.width - 40.0, hx=-1.0)
    assert agent.sense(_world())[6] == 0.0


def test_energy_input_is_a_fraction_of_max() -> None:
    assert _agent().sense(_world())[7] == CFG.start_energy / CFG.max_energy


# -- actions ---------------------------------------------------------------


def test_positive_steer_turns_clockwise_on_screen() -> None:
    agent = _agent()
    agent.steer, agent.throttle = 1.0, 0.0
    agent.act(_world(), Rng(0))
    assert agent.hy > 0
    assert math.atan2(agent.hy, agent.hx) == pytest.approx(2 * math.atan(CFG.turn_rate))


def test_heading_stays_a_unit_vector() -> None:
    agent = _agent()
    world, rng = _world(), Rng(0)
    for i in range(5000):
        agent.steer = 0.9 if i % 7 else -0.4
        agent.throttle = 0.0
        agent.energy = CFG.max_energy
        agent.act(world, rng)
    assert math.hypot(agent.hx, agent.hy) == pytest.approx(1.0, abs=1e-12)


def test_throttle_sets_speed_and_movement_costs_energy() -> None:
    agent = _agent()
    agent.steer, agent.throttle = 0.0, 1.0
    agent.act(_world(), Rng(0))
    assert agent.x == pytest.approx(480.0 + CFG.max_speed)
    cost = CFG.idle_cost + CFG.move_cost * CFG.max_speed**2
    assert agent.energy == pytest.approx(CFG.start_energy - cost)


def test_agents_stay_inside_the_walls() -> None:
    agent = _agent(x=CFG.width - 1.0)
    agent.steer, agent.throttle = 0.0, 1.0
    agent.act(_world(), Rng(0))
    assert agent.x == CFG.width - CFG.agent_radius


def test_eating_scores_restores_energy_and_respawns_the_food() -> None:
    world = World(Config(food_count=1, hazard_count=0), Rng(0))
    world.food_x, world.food_y = [482.0], [300.0]
    agent = _agent()
    agent.energy = 10.0
    agent.steer = agent.throttle = 0.0
    agent.act(world, Rng(1))
    assert agent.food_eaten == 1
    assert agent.energy == pytest.approx(10.0 - CFG.idle_cost + CFG.food_energy)
    assert (world.food_x[0], world.food_y[0]) != (482.0, 300.0)


def test_energy_is_capped() -> None:
    world = World(Config(food_count=1, hazard_count=0), Rng(0))
    world.food_x, world.food_y = [482.0], [300.0]
    agent = _agent()
    agent.energy = CFG.max_energy - 1
    agent.steer = agent.throttle = 0.0
    agent.act(world, Rng(1))
    assert agent.energy == CFG.max_energy


def test_food_cannot_be_farmed_by_standing_still() -> None:
    # The original simulator scored +1 per tick for parking on a food node.
    world = World(Config(food_count=1, hazard_count=0), Rng(0))
    world.food_x, world.food_y = [482.0], [300.0]
    agent = _agent()
    rng = Rng(1)
    for _ in range(300):
        agent.steer = agent.throttle = 0.0
        agent.act(world, rng)
    assert agent.food_eaten <= 1


def test_touching_a_hazard_hurts_and_is_counted() -> None:
    agent = _agent()
    agent.steer = agent.throttle = 0.0
    agent.act(_world(hazards=[(490.0, 300.0)]), Rng(0))
    assert agent.hazard_steps == 1
    assert agent.energy == pytest.approx(CFG.start_energy - CFG.idle_cost - CFG.hazard_damage)


def test_running_out_of_energy_is_fatal() -> None:
    agent = _agent()
    agent.energy = 0.01
    agent.steer = agent.throttle = 0.0
    agent.act(_world(), Rng(0))
    assert not agent.alive
    assert agent.energy == 0.0


def test_spawn_points_and_headings_are_valid() -> None:
    cfg = Config()
    rng = Rng(6)
    world = World(cfg, rng)
    layers = layer_sizes(cfg.hidden)
    for _ in range(200):
        a = Agent.spawn(Brain.random(layers, rng), world, rng)
        assert 0 < a.x < cfg.width
        assert 0 < a.y < cfg.height
        assert math.hypot(a.hx, a.hy) == pytest.approx(1.0)
