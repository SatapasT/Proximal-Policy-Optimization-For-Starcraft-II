from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import json

import numpy as np
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback


# -----------------------------
# EpisodeStatsWrapper (generic)
# -----------------------------

@dataclass
class _EpisodeTracker:
    ep_return: float = 0.0
    episodes_seen: int = 0
    action_counts: List[int] = field(default_factory=list)
    action_total_steps: int = 0


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Generic episode stats wrapper.

    Universal fields expected from env info (optional):
      - objective_left
      - objective_killed
      - objective_total
      - win (0/1)

    Still supports legacy kill_keys fallback.

    On episode end:
      info["episode_extra"] = {
        "ep_return", "episodes_seen", "score_total",
        "action_total_steps", "action_counts", "action_pcts",
        "killed_total",
        "objective_left", "objective_killed", "objective_total",
        "win"
      }
    """

    def __init__(self, env: gym.Env, num_action_types: int, kill_keys: Sequence[str] = ()):
        super().__init__(env)
        self.num_action_types = int(num_action_types)
        self.kill_keys = list(kill_keys)
        self._tracker = _EpisodeTracker(action_counts=[0] * self.num_action_types)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._tracker.ep_return = 0.0
        self._tracker.action_counts = [0] * self.num_action_types
        self._tracker.action_total_steps = 0
        return obs, info

    def step(self, action):
        # action counts: assumes MultiDiscrete([action_type, ...])
        try:
            a0 = int(action[0])
            if 0 <= a0 < self.num_action_types:
                self._tracker.action_counts[a0] += 1
        except Exception:
            pass
        self._tracker.action_total_steps += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._tracker.ep_return += float(reward)

        done = bool(terminated or truncated)
        if done:
            self._tracker.episodes_seen += 1

            info = dict(info) if isinstance(info, dict) else {}

            score_total = float(info.get("score_total", 0.0))
            total_steps = max(int(self._tracker.action_total_steps), 1)
            pcts = [c / total_steps for c in self._tracker.action_counts]

            # universal objective fields (may be None)
            objective_left = info.get("objective_left", None)
            objective_killed = info.get("objective_killed", None)
            objective_total = info.get("objective_total", None)
            win = info.get("win", None)

            # killed_total: prefer universal objective_killed, fallback to kill_keys
            killed_total = None
            if objective_killed is not None:
                try:
                    killed_total = int(objective_killed)
                except Exception:
                    killed_total = None

            if killed_total is None:
                ksum = 0
                if self.kill_keys:
                    for k in self.kill_keys:
                        try:
                            ksum += int(info.get(k, 0))
                        except Exception:
                            pass
                killed_total = int(ksum)

            info["episode_extra"] = {
                "ep_return": float(self._tracker.ep_return),
                "episodes_seen": int(self._tracker.episodes_seen),
                "score_total": score_total,
                "action_total_steps": int(total_steps),
                "action_counts": list(self._tracker.action_counts),
                "action_pcts": list(pcts),
                "killed_total": int(killed_total),
                "objective_left": objective_left,
                "objective_killed": objective_killed,
                "objective_total": objective_total,
                "win": win,
            }

        return obs, reward, terminated, truncated, info


# ---------------------------------------
# PrintEpisodeExtrasCallback (generic)
# ---------------------------------------

class PrintEpisodeExtrasCallback(BaseCallback):
    """
    - JSONL: per-episode rows (type="episode") every episode (or every N)
    - CSV (txt): summary rows only, header written once
    - JSONL: summary rows (type="summary") every print_every episodes
    - Console: prints ONLY on summary (every print_every)

    Optional:
      - win_rate_window: if provided and win is present, logs win-rate over that window.
    """

    def __init__(
        self,
        action_type_names: List[str],
        print_every: int = 50,
        log_dir: str | Path = "checkpoints",
        save_every_episodes: int = 500,
        save_dir: str | Path = "checkpoints",
        verbose: int = 1,
        # knobs
        log_episode_json_every: int = 1,   # 1 = every episode
        write_summary_json: bool = True,
        print_summary: bool = True,
        win_rate_window: int | None = None,
    ):
        super().__init__(verbose)

        self.action_type_names = list(action_type_names)
        self.num_action_types = len(self.action_type_names)

        self.print_every = int(max(1, print_every))
        self.save_every_episodes = int(save_every_episodes)

        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.log_episode_json_every = int(max(1, log_episode_json_every))
        self.write_summary_json = bool(write_summary_json)
        self.print_summary = bool(print_summary)

        self.win_rate_window = None if win_rate_window is None else int(max(1, win_rate_window))

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.csv_path = self.log_dir / f"train_log_{ts}.csv"
        self.jsonl_path = self.log_dir / f"train_metrics_{ts}.jsonl"

        self.ep = 0
        self.scores: list[float] = []
        self.returns: list[float] = []
        self.ep_lens: list[int] = []
        self.killed_total_hist: list[int] = []
        self.action_entropy_hist: list[float] = []
        self.action_pcts_hist: list[list[float]] = []
        self.win_hist: list[int] = []  # stores 0/1 only when present

        self._write_csv_header()

        if self.verbose:
            print(f"[LOG] Summary CSV -> {self.csv_path}")
            print(f"[LOG] JSONL -> {self.jsonl_path}")

    # ---------- file I/O ----------

    def _append_csv_row(self, row: str) -> None:
        with self.csv_path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")

    def _append_json(self, obj: dict) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _write_csv_header(self) -> None:
        meta = f"# run_start={datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} print_every={self.print_every}"
        header = (
            "ep,timesteps,mean_score,std_score,mean_return,std_return,"
            "mean_killed,std_killed,mean_len,std_len,mean_action_entropy"
        )
        if self.win_rate_window is not None:
            header += f",win_rate_{self.win_rate_window}"
        with self.csv_path.open("w", encoding="utf-8") as f:
            f.write(meta + "\n")
            f.write(header + "\n")

    # ---------- helpers ----------

    @staticmethod
    def _safe(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_").replace("-", "_").lower()

    @staticmethod
    def _action_entropy(pcts: list[float]) -> float:
        p = np.asarray(pcts, dtype=np.float32)
        p = np.clip(p, 1e-8, 1.0)
        s = float(p.sum())
        if s <= 0:
            return 0.0
        p = p / s
        return float(-np.sum(p * np.log(p)))

    def _save_checkpoint_ep(self) -> None:
        if self.model is None:
            return
        out = self.save_dir / f"ppo_ep{self.ep:06d}"
        self.model.save(str(out))
        if self.verbose:
            print(f"[CHECKPOINT] saved -> {out}.zip")

    # ---------- main callback ----------

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            extra = info.get("episode_extra", None) if isinstance(info, dict) else None
            if extra is None:
                continue

            self.ep += 1

            score = float(extra.get("score_total", 0.0))
            ret = float(extra.get("ep_return", 0.0))
            ep_len = int(extra.get("action_total_steps", 0))
            killed_total = int(extra.get("killed_total", 0))

            # win handling (optional)
            win = extra.get("win", None)
            if win in (0, 1):
                self.win_hist.append(int(win))

            pcts = extra.get("action_pcts", [0.0] * self.num_action_types)
            if not isinstance(pcts, (list, tuple)) or len(pcts) != self.num_action_types:
                pcts = [0.0] * self.num_action_types
            pcts = [float(x) for x in pcts]
            act_ent = self._action_entropy(pcts)

            self.scores.append(score)
            self.returns.append(ret)
            self.ep_lens.append(ep_len)
            self.killed_total_hist.append(killed_total)
            self.action_entropy_hist.append(act_ent)
            self.action_pcts_hist.append(list(pcts))

            # TB logging per-episode
            self.logger.record("custom/score_total", score)
            self.logger.record("custom/ep_return", ret)
            self.logger.record("custom/ep_len", float(ep_len))
            self.logger.record("custom/action_entropy", act_ent)
            self.logger.record("custom/killed_total", float(killed_total))
            if win in (0, 1):
                self.logger.record("custom/win", float(win))

            for i, name in enumerate(self.action_type_names):
                self.logger.record(f"actions/pct_{self._safe(name)}", float(pcts[i]))

            # JSONL per-episode row
            if (self.ep % self.log_episode_json_every) == 0:
                action_pcts_dict = {self._safe(n): float(pcts[i]) for i, n in enumerate(self.action_type_names)}
                self._append_json({
                    "type": "episode",
                    "ep": int(self.ep),
                    "timesteps": int(getattr(self, "num_timesteps", 0)),
                    "score_total": score,
                    "ep_return": ret,
                    "ep_len": int(ep_len),
                    "action_entropy": act_ent,
                    "killed_total": int(killed_total),
                    "win": int(win) if win in (0, 1) else None,
                    "action_pcts": action_pcts_dict,
                })

            # SUMMARY block (every print_every)
            if (self.ep % self.print_every) == 0:
                score_w = np.asarray(self.scores[-self.print_every:], dtype=np.float32)
                ret_w = np.asarray(self.returns[-self.print_every:], dtype=np.float32)
                len_w = np.asarray(self.ep_lens[-self.print_every:], dtype=np.float32)
                kill_w = np.asarray(self.killed_total_hist[-self.print_every:], dtype=np.float32)
                ent_w = np.asarray(self.action_entropy_hist[-self.print_every:], dtype=np.float32)

                mean_score = float(score_w.mean()) if score_w.size else 0.0
                std_score = float(score_w.std()) if score_w.size else 0.0
                mean_ret = float(ret_w.mean()) if ret_w.size else 0.0
                std_ret = float(ret_w.std()) if ret_w.size else 0.0
                mean_len = float(len_w.mean()) if len_w.size else 0.0
                std_len = float(len_w.std()) if len_w.size else 0.0
                mean_killed = float(kill_w.mean()) if kill_w.size else 0.0
                std_killed = float(kill_w.std()) if kill_w.size else 0.0
                mean_ent = float(ent_w.mean()) if ent_w.size else 0.0

                win_rate = None
                if self.win_rate_window is not None and len(self.win_hist) > 0:
                    w = self.win_hist[-self.win_rate_window:]
                    win_rate = float(np.mean(np.asarray(w, dtype=np.float32)))
                    self.logger.record(f"custom/win_rate_{self.win_rate_window}", win_rate)

                # TB summary logging
                self.logger.record(f"custom/mean_score_{self.print_every}", mean_score)
                self.logger.record(f"custom/mean_return_{self.print_every}", mean_ret)
                self.logger.record(f"custom/mean_len_{self.print_every}", mean_len)
                self.logger.record(f"custom/mean_killed_{self.print_every}", mean_killed)
                self.logger.record(f"custom/mean_entropy_{self.print_every}", mean_ent)

                # CSV summary row
                csv_row = (
                    f"{self.ep},{int(getattr(self,'num_timesteps',0))},"
                    f"{mean_score:.4f},{std_score:.4f},"
                    f"{mean_ret:.4f},{std_ret:.4f},"
                    f"{mean_killed:.4f},{std_killed:.4f},"
                    f"{mean_len:.4f},{std_len:.4f},"
                    f"{mean_ent:.6f}"
                )
                if self.win_rate_window is not None:
                    csv_row += f",{(win_rate if win_rate is not None else float('nan')):.4f}"
                self._append_csv_row(csv_row)

                # JSONL summary row (optional)
                if self.write_summary_json:
                    self._append_json({
                        "type": "summary",
                        "ep": int(self.ep),
                        "timesteps": int(getattr(self, "num_timesteps", 0)),
                        "mean_score": mean_score,
                        "std_score": std_score,
                        "mean_return": mean_ret,
                        "std_return": std_ret,
                        "mean_len": mean_len,
                        "std_len": std_len,
                        "mean_killed": mean_killed,
                        "std_killed": std_killed,
                        "mean_action_entropy": mean_ent,
                        "win_rate": win_rate,
                        "print_every": int(self.print_every),
                    })

                # console print ONLY when summary logs
                if self.print_summary:
                    msg = (
                        f"[summary] ep={self.ep} ts={int(getattr(self,'num_timesteps',0))} "
                        f"mean_score={mean_score:.2f}±{std_score:.2f} "
                        f"mean_return={mean_ret:.2f}±{std_ret:.2f} "
                        f"mean_len={mean_len:.1f}±{std_len:.1f} "
                        f"mean_killed={mean_killed:.2f}±{std_killed:.2f} "
                        f"ent={mean_ent:.3f}"
                    )
                    if win_rate is not None:
                        msg += f" win_rate_{self.win_rate_window}={win_rate:.2f}"
                    print(msg)

            # checkpointing
            if self.save_every_episodes > 0 and (self.ep % self.save_every_episodes == 0):
                self._save_checkpoint_ep()

        return True
