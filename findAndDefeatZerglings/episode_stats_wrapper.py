from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import gymnasium as gym


@dataclass
class _EpisodeTracker:
    ep_return: float = 0.0
    episodes_seen: int = 0
    action_counts: List[int] = field(default_factory=list)
    action_total_steps: int = 0


class EpisodeStatsWrapper(gym.Wrapper):
    """
    Tracks per-episode return and action distribution.

    On episode end, adds `info["episode_extra"]` with return/score/win and action stats.
    """

    def __init__(self, env: gym.Env, num_action_types: int):
        super().__init__(env)
        self.num_action_types = int(num_action_types)
        self._tracker = _EpisodeTracker(action_counts=[0] * self.num_action_types)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._tracker.ep_return = 0.0
        self._tracker.action_counts = [0] * self.num_action_types
        self._tracker.action_total_steps = 0
        return obs, info

    def step(self, action):
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

            score_total = float(info.get("score_total", 0.0))
            win = info.get("win", None)
            try:
                win_int = int(win) if win is not None else 0
            except Exception:
                win_int = 0

            total = max(int(self._tracker.action_total_steps), 1)
            pcts = [c / total for c in self._tracker.action_counts]

            info = dict(info)
            info["episode_extra"] = {
                "ep_return": float(self._tracker.ep_return),
                "episodes_seen": int(self._tracker.episodes_seen),
                "score_total": score_total,
                "win": win_int,
                "action_total_steps": total,
                "action_counts": list(self._tracker.action_counts),
                "action_pcts": list(pcts),
            }

        return obs, reward, terminated, truncated, info
