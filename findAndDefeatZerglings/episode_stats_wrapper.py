# episode_stats_wrapper.py
from __future__ import annotations
import gymnasium as gym


class EpisodeStatsWrapper(gym.Wrapper):
    def __init__(self, env, replay_every_ep: int = 350, max_replays: int = 10):
        super().__init__(env)
        self.replay_every_ep = int(replay_every_ep)
        self.max_replays = int(max_replays)

        self.episodes_seen = 0
        self.replays_saved = 0

        self.reset_episode_stats()

    def reset_episode_stats(self):
        self.ep_return = 0.0
        self.ep_len = 0
        self.last_score_total = 0
        self.last_zerglings_left = None
        self.last_win = 0  # keep as 0 unless proven win=1

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_episode_stats()
        return obs, info

    def save_replay(self):
        if hasattr(self.env, "save_replay"):
            return self.env.save_replay()
        return None

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.ep_return += float(reward)
        self.ep_len += 1

        if "score_total" in info:
            self.last_score_total = int(info["score_total"])
        if "zerglings_left" in info:
            self.last_zerglings_left = info["zerglings_left"]
        if "win" in info and info["win"] is not None:
            self.last_win = int(info["win"])

        done = bool(terminated or truncated)
        if done:
            self.episodes_seen += 1

            info = dict(info)
            info["episode_extra"] = {
                "ep_return": self.ep_return,
                "ep_len": self.ep_len,
                "score_total": self.last_score_total,
                "zerglings_left": self.last_zerglings_left,
                "win": self.last_win,
                "episodes_seen": self.episodes_seen,
                "last_win": self.last_win,
            }

            should_save = (
                self.replays_saved < self.max_replays
                and self.replay_every_ep > 0
                and (self.episodes_seen % self.replay_every_ep == 0)
            )
            info["save_replay"] = bool(should_save)
            if should_save:
                self.replays_saved += 1

        return obs, reward, terminated, truncated, info
