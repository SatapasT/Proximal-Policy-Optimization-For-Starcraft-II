import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pysc2.env import sc2_env
from pysc2.lib import actions, features


class FindAndDefeatZerglingsGym(gym.Env):
    """Gymnasium wrapper around PySC2's FindAndDefeatZerglings mini-game.

    Action space:
        - (action_type, target_cell) where target_cell is a coarse grid index.
    Observation:
        - concatenated feature_screen + resized feature_minimap, normalized per-layer.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        map_name: str = "FindAndDefeatZerglings",
        screen_size: int = 84,
        minimap_size: int = 64,
        grid_n: int = 8,
        step_mul: int = 8,
        visualize: bool = False,
    ):
        super().__init__()
        self.map_name = map_name
        self.screen_size = int(screen_size)
        self.minimap_size = int(minimap_size)
        self.grid_n = int(grid_n)
        self.step_mul = int(step_mul)
        self.visualize = bool(visualize)

        self.interface_format = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=(self.screen_size, self.screen_size),
                minimap=(self.minimap_size, self.minimap_size),
            ),
            use_feature_units=True,
            use_raw_units=False,
        )

        self.action_type_names = [
            "no_op",  # do nothing
            "select_army",  # select all army (F2)
            "select_point",  # click-select unit(s) at a screen point
            "select_rect",  # box-select units (drag rectangle)
            "move_camera",  # pan camera via minimap click
            "move_screen",  # issue Move command to a screen position
            "smart_screen",  # context action (right-click / Smart)
            "attack_screen",  # attack-move / attack target on screen
            "stop_quick",  # stop current orders
            "holdposition_quick",  # hold position (no chasing)
        ]
        self.num_action_types = len(self.action_type_names)
        self.action_space = spaces.MultiDiscrete([self.num_action_types, self.grid_n * self.grid_n])

        self.n_screen = len(features.SCREEN_FEATURES)
        self.n_minimap = len(features.MINIMAP_FEATURES)
        self.obs_channels = self.n_screen + self.n_minimap
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_channels, self.screen_size, self.screen_size),
            dtype=np.float32,
        )

        self._env = None
        self._last_ts = None
        self.fn: dict[str, int] = {}

        self.ZERGLING_TYPE_ID = 105

    def _launch(self):
        if self._env is not None:
            return

        self._env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.terran),
                sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy),
            ],
            agent_interface_format=self.interface_format,
            step_mul=self.step_mul,
            game_steps_per_episode=0,
            visualize=self.visualize,
            save_replay_episodes=0,
        )

        f = actions.FUNCTIONS
        self.fn = {
            "no_op": f.no_op.id,  # Do nothing
            "select_army": f.select_army.id,  # Select all army (equivalent to F2)
            "select_point": f.select_point.id,  # Select unit(s) at a screen point
            "select_rect": f.select_rect.id,  # Box-select units in a rectangle
            "move_camera": f.move_camera.id,  # Pan camera via minimap click
            "move_screen": f.Move_screen.id,  # Issue Move command to a screen position
            "smart_screen": f.Smart_screen.id,  # Context action (right-click / Smart)
            "attack_screen": f.Attack_screen.id,  # Attack-move / attack target on screen
            "stop_quick": f.Stop_quick.id,  # Stop current unit orders
            "holdposition_quick": f.HoldPosition_quick.id,  # Hold position (do not chase)
        }

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._launch()
        ts = self._env.reset()[0]
        self._last_ts = ts
        return self._obs_from_ts(ts), {}

    def step(self, action):
        action_type_idx = int(action[0])
        target_idx = int(action[1])

        sc2_action = self._make_sc2_action(action_type_idx, target_idx, self._last_ts)

        ts = self._env.step([sc2_action])[0]
        self._last_ts = ts

        obs = self._obs_from_ts(ts)
        reward = float(ts.reward) if ts.reward is not None else 0.0
        terminated = bool(ts.last())
        truncated = False

        o = ts.observation

        score_total = 0
        score_vec = o.get("score_cumulative", None)
        if score_vec is not None:
            score_vec_np = np.asarray(score_vec, dtype=np.int32)
            score_total = int(score_vec_np.max()) if score_vec_np.size > 0 else 0

        score_by_vital = o.get("score_by_vital", None)
        if score_total == 0 and score_by_vital is not None:
            sv = np.asarray(score_by_vital, dtype=np.int32)
            if sv.size > 0:
                score_total = int(sv.max())

        zerglings_left = None
        f_units = o.get("feature_units", None)
        if f_units is not None and len(f_units) > 0:
            arr = np.asarray(f_units)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                unit_type_col = 0
                alliance_col = 1
                is_zergling = (arr[:, unit_type_col] == self.ZERGLING_TYPE_ID)
                is_enemy = (arr[:, alliance_col] != 1)
                zerglings_left = int(np.sum(is_zergling & is_enemy))

        win = None
        if terminated and zerglings_left is not None:
            win = 1 if zerglings_left == 0 else 0

        info = {
            "game_loop": int(o.get("game_loop", 0)),
            "available_actions": o.get("available_actions", []),
            "score_total": score_total,
            "win": win,
        }

        return obs, reward, terminated, truncated, info

    def _obs_from_ts(self, ts):
        screen = ts.observation["feature_screen"].astype(np.float32)
        minimap = ts.observation["feature_minimap"].astype(np.float32)
        minimap_rs = self._nearest_resize(minimap, self.screen_size, self.screen_size)

        screen = self._per_layer_normalize(screen)
        minimap_rs = self._per_layer_normalize(minimap_rs)
        return np.concatenate([screen, minimap_rs], axis=0)


    @staticmethod
    def _per_layer_normalize(x: np.ndarray) -> np.ndarray:
        mx = np.maximum(x.max(axis=(1, 2), keepdims=True), 1.0)
        return x / mx

    @staticmethod
    def _nearest_resize(x: np.ndarray, H: int, W: int) -> np.ndarray:
        C, h, w = x.shape
        ys = (np.arange(H) * (h / H)).astype(int)
        xs = (np.arange(W) * (w / W)).astype(int)
        ys = np.clip(ys, 0, h - 1)
        xs = np.clip(xs, 0, w - 1)
        return x[:, ys[:, None], xs[None, :]]

    def _cell_to_xy(self, idx: int, size: int) -> list[int]:
        n = self.grid_n
        r = idx // n
        c = idx % n
        cell = size / n
        x = int((c + 0.5) * cell)
        y = int((r + 0.5) * cell)
        x = int(np.clip(x, 0, size - 1))
        y = int(np.clip(y, 0, size - 1))
        return [x, y]

    def _rect_around_xy(self, xy: list[int], box_px: int = 12) -> list[list[int]]:
        x, y = xy
        x0 = int(np.clip(x - box_px // 2, 0, self.screen_size - 1))
        y0 = int(np.clip(y - box_px // 2, 0, self.screen_size - 1))
        x1 = int(np.clip(x + box_px // 2, 0, self.screen_size - 1))
        y1 = int(np.clip(y + box_px // 2, 0, self.screen_size - 1))
        return [[x0, y0], [x1, y1]]

    def _make_sc2_action(self, action_type_idx: int, target_idx: int, last_ts):
        avail = set(last_ts.observation.get("available_actions", []))

        name = self.action_type_names[action_type_idx]
        fn_id = self.fn[name]

        if fn_id not in avail:
            return actions.FunctionCall(self.fn["no_op"], [])

        if name == "no_op":
            return actions.FunctionCall(fn_id, [])
        if name == "select_army":
            return actions.FunctionCall(fn_id, [[0]])
        if name in ["stop_quick", "holdposition_quick"]:
            return actions.FunctionCall(fn_id, [[0]])

        if name == "move_camera":
            xy = self._cell_to_xy(target_idx, self.minimap_size)
            return actions.FunctionCall(fn_id, [xy])

        xy = self._cell_to_xy(target_idx, self.screen_size)

        if name == "select_point":
            return actions.FunctionCall(fn_id, [[0], xy])

        if name == "select_rect":
            rect = self._rect_around_xy(xy, box_px=12)
            return actions.FunctionCall(fn_id, [[0], rect[0], rect[1]])

        if name in ["move_screen", "smart_screen", "attack_screen"]:
            return actions.FunctionCall(fn_id, [[0], xy])

        return actions.FunctionCall(self.fn["no_op"], [])
