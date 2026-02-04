from __future__ import annotations

from pysc2.lib import units
from common.sc2_base_env import BaseSC2Gym


class DefeatRoachesGym(BaseSC2Gym):
    def __init__(
        self,
        map_name: str = "DefeatRoaches",
        screen_size: int = 84,
        minimap_size: int = 64,
        grid_n: int = 8,
        step_mul: int = 8,
        visualize: bool = False,
        camera_grid_n: int = 4,
        camera_cooldown: int = 6,
    ):
        super().__init__(
            map_name=map_name,
            screen_size=screen_size,
            minimap_size=minimap_size,
            grid_n=grid_n,
            step_mul=step_mul,
            visualize=visualize,
            camera_grid_n=camera_grid_n,
            camera_cooldown=camera_cooldown,
        )
        self.ROACH_TYPE_ID = int(units.Zerg.Roach)

    def _info_extra(self, ts, o: dict) -> dict:
        f_units = o.get("feature_units", None)
        roaches_left = self._count_enemy_types(f_units, [self.ROACH_TYPE_ID])

        win = None
        if ts.last() and roaches_left is not None:
            win = 1 if int(roaches_left) == 0 else 0

        return {
            "roaches_left": int(roaches_left) if roaches_left is not None else None,
            "win": win,
        }
