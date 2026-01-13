# sc2_gym.py  (restricted actions + metrics + robust replay copying)
from __future__ import annotations

import os
import time
import glob
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pysc2.env import sc2_env
from pysc2.lib import actions, features


def _default_sc2_replay_root() -> str:
    """
    Most consistent SC2 replay location on Windows.
    SC2 may still store under Accounts/<id>/Replays as well.
    """
    return str(Path.home() / "Documents" / "StarCraft II" / "Replays")


def _default_sc2_accounts_root() -> str:
    return str(Path.home() / "Documents" / "StarCraft II" / "Accounts")


def _list_replays_in_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    for p in paths:
        try:
            out.extend(glob.glob(os.path.join(p, "*.SC2Replay")))
        except Exception:
            pass
    # newest last by mtime
    out.sort(key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0)
    return out


class FindAndDefeatZerglingsGym(gym.Env):
    """
    Observation: concat(feature_screen, resized_feature_minimap) -> (C,84,84) float32 in [0,1]
    Action: MultiDiscrete([num_action_types, grid_n*grid_n])
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
        # Where SC2/PySC2 will try to write replays (SC2 may ignore this on Windows)
        sc2_replay_dir: Optional[str] = None,
        # Where YOU want the replays to end up (we copy here after saving)
        replay_out_dir: str = "sc2_replays",
    ):
        super().__init__()
        self.map_name = map_name
        self.screen_size = int(screen_size)
        self.minimap_size = int(minimap_size)
        self.grid_n = int(grid_n)
        self.step_mul = int(step_mul)
        self.visualize = bool(visualize)

        # 1) SC2/PySC2 replay folder (may be ignored)
        self.sc2_replay_dir = str(sc2_replay_dir) if sc2_replay_dir is not None else _default_sc2_replay_root()
        os.makedirs(self.sc2_replay_dir, exist_ok=True)

        # 2) Your project output folder (we copy here)
        self.replay_out_dir = str(replay_out_dir)
        os.makedirs(self.replay_out_dir, exist_ok=True)

        # For extra safety, also watch the SC2 Accounts/<id>/Replays folders
        self._accounts_root = _default_sc2_accounts_root()

        self.interface_format = features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(
                screen=(self.screen_size, self.screen_size),
                minimap=(self.minimap_size, self.minimap_size),
            ),
            use_feature_units=True,
            use_raw_units=False,
        )

        # Restrict to a small action set
        self.action_type_names = [
            "no_op",
            "select_army",
            "move_camera",        # minimap target
            "move_screen",        # screen target
            "move_minimap",       # minimap target
            "smart_screen",       # screen target
            "smart_minimap",      # minimap target
            "attack_screen",      # screen target
            "attack_minimap",     # minimap target
            "stop_quick",
            "holdposition_quick",
        ]
        self.num_action_types = len(self.action_type_names)
        self.action_space = spaces.MultiDiscrete([self.num_action_types, self.grid_n * self.grid_n])

        # Observation: all screen + all minimap channels (minimap resized to 84x84)
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

        # Unit type for Zergling (SC2 unit type id)
        self.ZERGLING_TYPE_ID = 105

    def _launch(self):
        if self._env is not None:
            return

        # NOTE: visualize=False means no live window rendering
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
            # Try to hint replay dir (SC2 may still ignore; we copy afterwards anyway)
            save_replay_episodes=0,
            replay_dir=self.sc2_replay_dir,
        )

        f = actions.FUNCTIONS
        self.fn = {
            "no_op": f.no_op.id,
            "select_army": f.select_army.id,
            "move_camera": f.move_camera.id,
            "move_screen": f.Move_screen.id,
            "move_minimap": f.Move_minimap.id,
            "smart_screen": f.Smart_screen.id,
            "smart_minimap": f.Smart_minimap.id,
            "attack_screen": f.Attack_screen.id,
            "attack_minimap": f.Attack_minimap.id,
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
        return self._obs_from_ts(ts), {}

    # ---------------- replay saving ----------------

    def _candidate_replay_paths(self) -> list[str]:
        paths = [self.sc2_replay_dir]

        # Also include Accounts/*/Replays because SC2 often writes there
        try:
            if os.path.isdir(self._accounts_root):
                acc = glob.glob(os.path.join(self._accounts_root, "*", "Replays"))
                paths.extend(acc)
        except Exception:
            pass

        # de-dup
        uniq = []
        seen = set()
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in seen:
                uniq.append(ap)
                seen.add(ap)
        return uniq

    def save_replay(self, prefix: Optional[str] = None) -> Optional[str]:
        """
        Save a replay via PySC2/SC2, then COPY it into replay_out_dir.
        Returns the copied replay path in replay_out_dir if successful, else None.
        """
        if self._env is None:
            return None

        if prefix is None:
            prefix = f"{self.map_name}_{int(time.time())}"

        candidates = self._candidate_replay_paths()
        before = set(_list_replays_in_paths(candidates))

        # Ask SC2/PySC2 to save
        try:
            try:
                # Newer pysc2 signature
                self._env.save_replay(replay_dir=self.sc2_replay_dir, prefix=prefix)
            except TypeError:
                # Older pysc2 signature
                self._env.save_replay()
        except Exception as e:
            print(f"[WARN] save_replay failed: {e}")
            return None

        # Give filesystem a moment on Windows
        time.sleep(0.25)

        after = set(_list_replays_in_paths(candidates))
        new_files = sorted(list(after - before), key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0)

        if not new_files:
            print("[REPLAY] save_replay called, but no new .SC2Replay detected.")
            print("[REPLAY] Look in:")
            for p in candidates:
                print("         ", p)
            return None

        src = new_files[-1]
        dst_name = os.path.basename(src)
        dst = os.path.join(self.replay_out_dir, dst_name)

        try:
            shutil.copy2(src, dst)
            print(f"[REPLAY] Saved by SC2 -> {src}")
            print(f"[REPLAY] Copied to    -> {dst}")
            return dst
        except Exception as e:
            print(f"[WARN] Could not copy replay into project folder: {e}")
            print(f"[REPLAY] Replay is at -> {src}")
            return None

    # ---------------- step ----------------

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

        # ---- score_total (robust proxy) ----
        score_total = 0
        score_vec_np = None
        score_vec = o.get("score_cumulative", None)
        if score_vec is not None:
            score_vec_np = np.asarray(score_vec, dtype=np.int32)
            score_total = int(score_vec_np.max()) if score_vec_np.size > 0 else 0

        score_by_vital = o.get("score_by_vital", None)
        if score_total == 0 and score_by_vital is not None:
            sv = np.asarray(score_by_vital, dtype=np.int32)
            if sv.size > 0:
                score_total = int(sv.max())

        # ---- zerglings left ----
        zerglings_left = None
        f_units = o.get("feature_units", None)
        if f_units is not None and len(f_units) > 0:
            arr = np.asarray(f_units)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                unit_type_col = 0
                alliance_col = 1
                is_zergling = (arr[:, unit_type_col] == self.ZERGLING_TYPE_ID)
                is_enemy = (arr[:, alliance_col] != 1)  # self usually 1
                zerglings_left = int(np.sum(is_zergling & is_enemy))

        # ---- win ----
        win = None
        if terminated and zerglings_left is not None:
            win = 1 if zerglings_left == 0 else 0

        info = {
            "game_loop": int(o.get("game_loop", 0)),
            "available_actions": o.get("available_actions", []),
            "score_total": score_total,
            "score_cumulative_vec": score_vec_np,  # can be None
            "zerglings_left": zerglings_left,
            "win": win,
        }

        return obs, reward, terminated, truncated, info

    # ---------------- observation ----------------

    def _obs_from_ts(self, ts):
        screen = ts.observation["feature_screen"].astype(np.float32)  # (C,84,84)
        minimap = ts.observation["feature_minimap"].astype(np.float32)  # (C,64,64)
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

    # ---------------- action mapping ----------------

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

    def _make_sc2_action(self, action_type_idx: int, target_idx: int, last_ts):
        avail = set(last_ts.observation.get("available_actions", []))

        name = self.action_type_names[action_type_idx]
        fn_id = self.fn[name]

        # If not available, try select_army if possible
        if fn_id not in avail:
            sel_id = self.fn["select_army"]
            if sel_id in avail:
                return actions.FunctionCall(sel_id, [[0]])
            return actions.FunctionCall(self.fn["no_op"], [])

        # Non-targeted
        if name in ["no_op", "stop_quick", "holdposition_quick"]:
            if name in ["stop_quick", "holdposition_quick"]:
                return actions.FunctionCall(fn_id, [[0]])  # queued=0
            return actions.FunctionCall(fn_id, [])

        if name == "select_army":
            return actions.FunctionCall(fn_id, [[0]])

        # Minimap-targeted
        if name in ["move_camera", "move_minimap", "smart_minimap", "attack_minimap"]:
            xy = self._cell_to_xy(target_idx, self.minimap_size)
            if name == "move_camera":
                return actions.FunctionCall(fn_id, [xy])
            return actions.FunctionCall(fn_id, [[0], xy])

        # Screen-targeted
        if name in ["move_screen", "smart_screen", "attack_screen"]:
            xy = self._cell_to_xy(target_idx, self.screen_size)
            return actions.FunctionCall(fn_id, [[0], xy])

        return actions.FunctionCall(self.fn["no_op"], [])
