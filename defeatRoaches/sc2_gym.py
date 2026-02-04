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

        # DefeatRoaches is fixed count in the minigame (you believe 4)
        self._objective_total = 4

        # per-episode trackers
        self._prev_left: int | None = None
        self._killed: int = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_left = None
        self._killed = 0
        return obs, info

    def _info_extra(self, ts, o: dict) -> dict:
        f_units = o.get("feature_units", None)

        left = self._count_enemy_types(f_units, [self.ROACH_TYPE_ID])

        # compute killed by tracking "left" deltas (robust if score doesn't expose kills)
        if left is not None:
            left_i = int(left)
            if self._prev_left is not None:
                d = self._prev_left - left_i
                if d > 0:
                    self._killed += int(d)
            self._prev_left = left_i

        win = None
        if ts.last() and left is not None:
            win = 1 if int(left) == 0 else 0

        # universal objective keys + minigame-specific keys
        return {
            # universal
            "objective_left": int(left) if left is not None else None,
            "objective_killed": int(self._killed) if left is not None else None,
            "objective_total": int(self._objective_total),
            "win": win,

            # minigame-specific
            "roaches_left": int(left) if left is not None else None,
            "roaches_killed": int(self._killed) if left is not None else None,
        }
