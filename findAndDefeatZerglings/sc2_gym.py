from pysc2.lib import units
from common.sc2_base_env import BaseSC2Gym


class FindAndDefeatZerglingsGym(BaseSC2Gym):
    def __init__(
        self,
        map_name: str = "FindAndDefeatZerglings",
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
        self._prev_left: int | None = None
        self._killed: int = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev_left = None
        self._killed = 0
        return obs, info

    def _info_extra(self, ts, o: dict) -> dict:
        f_units = o.get("feature_units", None)
        left = self._count_enemy_types(f_units, [self.ZERGLING_TYPE_ID])

        # update cumulative kills via left-deltas (monotonic)
        if left is not None:
            left_i = int(left)
            if self._prev_left is None:
                self._prev_left = left_i
            else:
                delta = int(self._prev_left) - left_i
                if delta > 0:
                    self._killed += int(delta)
                self._prev_left = left_i

        return {
            "zerglings_left": int(left) if left is not None else None,
            "zerglings_killed": int(self._killed),
        }
