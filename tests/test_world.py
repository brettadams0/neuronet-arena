from neuronet.config import Config
from neuronet.rng import Rng
from neuronet.world import World


def _clearance(cfg: Config) -> float:
    return cfg.hazard_radius + cfg.food_radius + 2 * cfg.agent_radius


def test_world_has_the_configured_counts() -> None:
    cfg = Config(food_count=17, hazard_count=5)
    world = World(cfg, Rng(1))
    assert len(world.food_x) == len(world.food_y) == 17
    assert len(world.hazard_x) == len(world.hazard_y) == 5


def test_everything_is_inside_the_arena() -> None:
    cfg = Config()
    world = World(cfg, Rng(2))
    for x, y in zip(world.food_x + world.hazard_x, world.food_y + world.hazard_y, strict=True):
        assert 0 < x < cfg.width
        assert 0 < y < cfg.height


def test_food_never_spawns_where_eating_it_means_touching_a_hazard() -> None:
    cfg = Config(hazard_count=20)
    for seed in range(20):
        world = World(cfg, Rng(seed))
        clear = _clearance(cfg)
        for fx, fy in zip(world.food_x, world.food_y, strict=True):
            for hx, hy in zip(world.hazard_x, world.hazard_y, strict=True):
                assert (fx - hx) ** 2 + (fy - hy) ** 2 >= clear**2


def test_respawn_moves_only_the_eaten_item() -> None:
    cfg = Config()
    rng = Rng(3)
    world = World(cfg, rng)
    before = list(zip(world.food_x, world.food_y, strict=True))
    world.respawn_food(4, rng)
    after = list(zip(world.food_x, world.food_y, strict=True))
    assert after[4] != before[4]
    assert after[:4] == before[:4]
    assert after[5:] == before[5:]


def test_same_seed_same_world() -> None:
    a, b = World(Config(), Rng(8)), World(Config(), Rng(8))
    assert (a.food_x, a.food_y) == (b.food_x, b.food_y)
    assert (a.hazard_x, a.hazard_y) == (b.hazard_x, b.hazard_y)
