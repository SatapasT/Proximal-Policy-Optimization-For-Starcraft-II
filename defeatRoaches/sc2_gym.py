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

        time_penalty: float = 0.01,      # stronger "stop stalling"
        kill_bonus: float = 1.0,          # was 0.25 in BaseSC2Gym default (big boost)
        own_loss_penalty: float = 0.25,   # was 0.5 default (slightly softer to avoid fear/stall)
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
            time_penalty=time_penalty,
            kill_bonus=kill_bonus,
            own_loss_penalty=own_loss_penalty,
        )

        self.ROACH_TYPE_ID = int(units.Zerg.Roach)

        self._objective_total: int | None = None
        self._prev_roaches_left: int | None = None
        self._roaches_killed: int = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_roaches_left = None
        self._roaches_killed = 0
        self._objective_total = None
        return obs, info

    def _info_extra(self, ts, o: dict) -> dict:
        f_units = o.get("feature_units", None)
        roaches_left = self._count_enemy_types(f_units, [self.ROACH_TYPE_ID])

        if roaches_left is not None:
            left_i = int(roaches_left)

            if self._prev_roaches_left is None:
                self._prev_roaches_left = left_i
                self._objective_total = left_i
            else:
                delta = int(self._prev_roaches_left) - left_i
                if delta > 0:
                    self._roaches_killed += int(delta)
                self._prev_roaches_left = left_i

        objective_left = int(roaches_left) if roaches_left is not None else None
        objective_killed = int(self._roaches_killed) if roaches_left is not None else None
        objective_total = int(self._objective_total) if self._objective_total is not None else None

        win = None
        if ts.last() and roaches_left is not None:
            win = 1 if int(roaches_left) == 0 else 0

        return {
            "objective_left": objective_left,
            "objective_killed": objective_killed,
            "objective_total": objective_total,
            "win": win,

            "roaches_left": objective_left,
            "roaches_killed": int(self._roaches_killed),
        }