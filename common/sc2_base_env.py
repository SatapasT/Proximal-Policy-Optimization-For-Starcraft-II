import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pysc2.env import sc2_env
from pysc2.lib import actions, features


class BaseSC2Gym(gym.Env):
    metadata = {"render_modes": []}

    # NOTE: select_army removed, select_own_unit added
    ACTION_TYPE_NAMES = [
        "no_op",
        "select_own_unit",
        "select_point",
        "select_rect",
        "move_camera",
        "move_screen",
        "smart_screen",
        "attack_screen",
        "stop_quick",
        "holdposition_quick",
    ]

    def __init__(
        self,
        map_name: str,
        screen_size: int = 84,
        minimap_size: int = 64,
        grid_n: int = 8,
        step_mul: int = 8,
        visualize: bool = False,
        camera_grid_n: int = 4,
        camera_cooldown: int = 6,
        players=None,
    ):
        super().__init__()
        self.map_name = map_name
        self.screen_size = int(screen_size)
        self.minimap_size = int(minimap_size)
        self.grid_n = int(grid_n)
        self.step_mul = int(step_mul)
        self.visualize = bool(visualize)

        self.camera_grid_n = max(2, min(int(camera_grid_n), 32))
        self.camera_cooldown = max(0, int(camera_cooldown))
        self._steps_since_cam = 10**9

        self.interface_format = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=(self.screen_size, self.screen_size),
                minimap=(self.minimap_size, self.minimap_size),
            ),
            use_feature_units=True,
            use_raw_units=False,
        )

        self.action_type_names = list(self.ACTION_TYPE_NAMES)
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

        # players can differ per minigame; default is 1 agent + zerg very_easy bot
        self.players = players or [
            sc2_env.Agent(sc2_env.Race.terran),
            sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.very_easy),
        ]

    def _launch(self):
        if self._env is not None:
            return

        self._env = sc2_env.SC2Env(
            map_name=self.map_name,
            players=self.players,
            agent_interface_format=self.interface_format,
            step_mul=self.step_mul,
            game_steps_per_episode=0,
            visualize=self.visualize,
            save_replay_episodes=0,
        )

        f = actions.FUNCTIONS
        # NOTE: select_army removed
        # NOTE: select_own_unit uses select_point function id
        self.fn = {
            "no_op": f.no_op.id,
            "select_own_unit": f.select_point.id,
            "select_point": f.select_point.id,
            "select_rect": f.select_rect.id,
            "move_camera": f.move_camera.id,
            "move_screen": f.Move_screen.id,
            "smart_screen": f.Smart_screen.id,
            "attack_screen": f.Attack_screen.id,
            "stop_quick": f.Stop_quick.id,
            "holdposition_quick": f.HoldPosition_quick.id,
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
        self._steps_since_cam = 10**9
        return self._obs_from_ts(ts), {}

    # ---------- obs ----------

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

    def _target_to_coarse_minimap_xy(self, target_idx: int) -> list[int]:
        n = self.grid_n
        r = target_idx // n
        c = target_idx % n
        u = (c + 0.5) / n
        v = (r + 0.5) / n

        cg = self.camera_grid_n
        cc = int(np.clip(u * cg, 0, cg - 1))
        rr = int(np.clip(v * cg, 0, cg - 1))

        cell = self.minimap_size / cg
        x = int((cc + 0.5) * cell)
        y = int((rr + 0.5) * cell)
        x = int(np.clip(x, 0, self.minimap_size - 1))
        y = int(np.clip(y, 0, self.minimap_size - 1))
        return [x, y]

    # ---------- unit selection helpers ----------

    @staticmethod
    def _get_feature_unit_fields(u):
        """Robustly get (alliance, x, y) from a FeatureUnit-like object."""
        # preferred: namedtuple attributes
        try:
            alliance = int(getattr(u, "alliance"))
            x = int(getattr(u, "x"))
            y = int(getattr(u, "y"))
            return alliance, x, y
        except Exception:
            pass

        # fallback: tuple indexing (best-effort)
        try:
            t = tuple(u)
            # pysc2 FeatureUnit layout usually has:
            # [0]=unit_type, [1]=alliance, ... and x,y later.
            # We'll try common placements; if it fails we return None.
            alliance = int(t[1])
            # try last two as x,y (works in some builds)
            x = int(t[-2])
            y = int(t[-1])
            return alliance, x, y
        except Exception:
            return None

    def _select_own_unit_action(self, target_idx: int, last_ts):
        """
        Deterministically select one of our units using target_idx.
        This makes unit selection learnable for PPO.

        Strategy:
          - filter feature_units with alliance == 1 (self)
          - sort by (y, x)
          - pick k = target_idx % len(units)
          - select_point on that unit's (x, y)
        """
        o = last_ts.observation
        f_units = o.get("feature_units", None)
        if f_units is None or len(f_units) == 0:
            return actions.FunctionCall(self.fn["no_op"], [])


        own = []
        for u in f_units:
            fields = self._get_feature_unit_fields(u)
            if fields is None:
                continue
            alliance, x, y = fields
            if alliance == 1:  # self
                # clamp to screen bounds
                x = int(np.clip(x, 0, self.screen_size - 1))
                y = int(np.clip(y, 0, self.screen_size - 1))
                own.append((y, x))

        if not own:
            return actions.FunctionCall(self.fn["no_op"], [])

        own.sort()  # sort by (y, x)
        k = int(target_idx) % len(own)
        y, x = own[k]
        xy = [int(x), int(y)]

        # select_point: [[select_type], [x,y]] with select_type 0
        return actions.FunctionCall(self.fn["select_point"], [[0], xy])

    # ---------- action mapping ----------

    def _make_sc2_action(self, action_type_idx: int, target_idx: int, last_ts):
        avail = set(last_ts.observation.get("available_actions", []))
        name = self.action_type_names[action_type_idx]
        fn_id = self.fn[name]

        self._steps_since_cam += 1

        if fn_id not in avail:
            return actions.FunctionCall(self.fn["no_op"], [])

        if name == "no_op":
            return actions.FunctionCall(fn_id, [])

        if name == "select_own_unit":
            return self._select_own_unit_action(target_idx, last_ts)

        if name in ("stop_quick", "holdposition_quick"):
            return actions.FunctionCall(fn_id, [[0]])

        if name == "move_camera":
            if self.camera_cooldown > 0 and self._steps_since_cam < self.camera_cooldown:
                return actions.FunctionCall(self.fn["no_op"], [])
            xy = self._target_to_coarse_minimap_xy(target_idx)
            self._steps_since_cam = 0
            return actions.FunctionCall(fn_id, [xy])

        xy = self._cell_to_xy(target_idx, self.screen_size)

        if name == "select_point":
            return actions.FunctionCall(fn_id, [[0], xy])

        if name == "select_rect":
            rect = self._rect_around_xy(xy, box_px=12)
            return actions.FunctionCall(fn_id, [[0], rect[0], rect[1]])

        if name in ("move_screen", "smart_screen", "attack_screen"):
            return actions.FunctionCall(fn_id, [[0], xy])

        return actions.FunctionCall(self.fn["no_op"], [])

    # ---------- scoring helpers ----------

    @staticmethod
    def _score_total_from_obs(o: dict) -> int:
        score_by_vital = o.get("score_by_vital", None)
        if score_by_vital is not None:
            try:
                sv = np.asarray(score_by_vital, dtype=np.int32)
                if sv.size > 0:
                    return int(sv.sum())
            except Exception:
                pass

        score_vec = o.get("score_cumulative", None)
        if score_vec is not None:
            try:
                v = np.asarray(score_vec, dtype=np.int32)
                if v.size > 0:
                    return int(v.sum())
            except Exception:
                pass

        return 0

    @staticmethod
    def _count_enemy_types(feature_units, type_ids: list[int]) -> int | None:
        if feature_units is None or len(feature_units) == 0:
            return None
        arr = np.asarray(feature_units)
        if arr.ndim != 2 or arr.shape[1] < 2:
            return None

        unit_type_col = 0
        alliance_col = 1
        is_enemy = (arr[:, alliance_col] == 4)
        is_type = np.zeros(arr.shape[0], dtype=bool)
        for tid in type_ids:
            is_type |= (arr[:, unit_type_col] == tid)
        return int(np.sum(is_enemy & is_type))

    def _info_extra(self, ts, o: dict) -> dict:
        return {}

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
        info = {
            "game_loop": int(o.get("game_loop", 0)),
            "available_actions": o.get("available_actions", []),
            "score_total": self._score_total_from_obs(o),
        }
        info.update(self._info_extra(ts, o))
        return obs, reward, terminated, truncated, info
