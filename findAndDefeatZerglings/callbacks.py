from __future__ import annotations

from pathlib import Path
from typing import List
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PrintEpisodeExtrasCallback(BaseCallback):
    """
        Consumes `info["episode_extra"]` (from EpisodeStatsWrapper) and:
            - prints/writes a short summary every `print_every` episodes
            - records scalar metrics + action percentages to TensorBoard
            - saves a checkpoint every `save_every_episodes` episodes
    """

    def __init__(
        self,
        action_type_names: List[str],
        print_every: int = 20,
        win_rate_window: int = 10,
        log_dir: str | Path = "checkpoints",
        save_every_episodes: int = 500,
        save_dir: str | Path = "checkpoints",
        verbose: int = 1,
    ):
        super().__init__(verbose)

        self.action_type_names = list(action_type_names)
        self.num_action_types = len(self.action_type_names)

        self.print_every = int(print_every)
        self.win_rate_window = int(win_rate_window)

        self.save_every_episodes = int(save_every_episodes)
        self.save_dir = Path(save_dir)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = self.log_dir / f"train_log_{ts}.txt"

        self.ep = 0
        self.scores: list[float] = []
        self.returns: list[float] = []
        self.wins: list[int] = []
        self.action_pcts_hist: list[list[float]] = []

        if self.verbose:
            print(f"[LOG] Writing custom summaries to {self.log_path}")

    def _append(self, line: str) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    @staticmethod
    def _safe(name: str) -> str:
        return name.replace("/", "_").replace(" ", "_").replace("-", "_").lower()

    def _save_checkpoint_ep(self) -> None:
        if self.model is None:
            return
        out = self.save_dir / f"ppo_ep{self.ep:06d}"
        self.model.save(str(out))
        if self.verbose:
            print(f"[CHECKPOINT] saved -> {out}.zip")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            extra = info.get("episode_extra", None)
            if extra is None:
                continue

            self.ep += 1

            score = float(extra.get("score_total", 0.0))
            ret = float(extra.get("ep_return", 0.0))
            win = int(extra.get("win", 0))

            self.scores.append(score)
            self.returns.append(ret)
            self.wins.append(win)

            pcts = extra.get("action_pcts", [0.0] * self.num_action_types)
            if not isinstance(pcts, (list, tuple)) or len(pcts) != self.num_action_types:
                pcts = [0.0] * self.num_action_types
            pcts = [float(x) for x in pcts]
            self.action_pcts_hist.append(list(pcts))

            w = min(len(self.wins), self.win_rate_window)
            win_rate = float(np.mean(self.wins[-w:])) if w > 0 else 0.0

            self.logger.record("custom/score_total", score)
            self.logger.record("custom/ep_return", ret)
            self.logger.record(f"custom/win_rate_{self.win_rate_window}", win_rate)
            for i, name in enumerate(self.action_type_names):
                self.logger.record(f"actions/pct_{self._safe(name)}", float(pcts[i]))

            if self.ep % self.print_every == 0:
                mean_score = float(np.mean(self.scores[-self.print_every:]))
                mean_ret = float(np.mean(self.returns[-self.print_every:]))

                line = (
                    f"[episodes={self.ep}] "
                    f"mean_score={mean_score:.2f} "
                    f"mean_return={mean_ret:.2f} "
                    f"win_rate={win_rate:.2f}"
                )
                print(line)
                self._append(line)

                last = np.array(self.action_pcts_hist[-self.print_every:], dtype=np.float32)
                mean_pcts = last.mean(axis=0) if last.size else np.zeros(self.num_action_types, dtype=np.float32)

                parts = [f"{name}={mean_pcts[i]*100:.1f}%" for i, name in enumerate(self.action_type_names)]
                action_line = "[actions] " + ", ".join(parts)
                print(action_line)
                self._append(action_line)

            if self.save_every_episodes > 0 and (self.ep % self.save_every_episodes == 0):
                self._save_checkpoint_ep()

        return True
