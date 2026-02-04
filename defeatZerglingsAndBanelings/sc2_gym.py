from __future__ import annotations

from pysc2.lib import units
from common.sc2_base_env import BaseSC2Gym


class DefeatZerglingsAndBanelingsGym(BaseSC2Gym):
    def __init__(
        self,
        map_name: str = "DefeatZerglingsAndBanelings",
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

        self.ZERGLING_TYPE_ID = int(units.Zerg.Zergling)
        self.BANELING_TYPE_ID = int(units.Zerg.Baneling)

        self._objective_total: int | None = None

        self._prev_zerg_left: int | None = None
        self._prev_bane_left: int | None = None
        self._zerg_killed: int = 0
        self._bane_killed: int = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_zerg_left = None
        self._prev_bane_left = None
        self._zerg_killed = 0
        self._bane_killed = 0
        return obs, info

    def _info_extra(self, ts, o: dict) -> dict:
        f_units = o.get("feature_units", None)

        zerg_left = self._count_enemy_types(f_units, [self.ZERGLING_TYPE_ID])
        bane_left = self._count_enemy_types(f_units, [self.BANELING_TYPE_ID])

        if zerg_left is not None:
            z = int(zerg_left)
            if self._prev_zerg_left is not None:
                dz = self._prev_zerg_left - z
                if dz > 0:
                    self._zerg_killed += int(dz)
            self._prev_zerg_left = z

        if bane_left is not None:
            b = int(bane_left)
            if self._prev_bane_left is not None:
                db = self._prev_bane_left - b
                if db > 0:
                    self._bane_killed += int(db)
            self._prev_bane_left = b

        objective_left = None
        objective_killed = None
        if (zerg_left is not None) and (bane_left is not None):
            objective_left = int(zerg_left) + int(bane_left)
            objective_killed = int(self._zerg_killed) + int(self._bane_killed)

        win = None
        if ts.last() and objective_left is not None:
            win = 1 if int(objective_left) == 0 else 0

        return {
            "objective_left": objective_left,
            "objective_killed": objective_killed,
            "objective_total": self._objective_total,
            "win": win,

            "zerglings_left": int(zerg_left) if zerg_left is not None else None,
            "banelings_left": int(bane_left) if bane_left is not None else None,
            "zerglings_killed": int(self._zerg_killed) if zerg_left is not None else None,
            "banelings_killed": int(self._bane_killed) if bane_left is not None else None,
        }
