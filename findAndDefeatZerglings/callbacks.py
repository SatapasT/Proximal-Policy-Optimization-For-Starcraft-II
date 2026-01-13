# callbacks.py
from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PrintEpisodeExtrasCallback(BaseCallback):
    def __init__(
        self,
        print_every: int = 20,
        verbose: int = 1,
        save_on_first_win: bool = True,
        first_win_path: str = "checkpoints/ppo_first_win",
    ):
        super().__init__(verbose)
        self.print_every = int(print_every)

        self.save_on_first_win = bool(save_on_first_win)
        self.first_win_path = str(first_win_path)
        self._saved_first_win = False

        self.ep = 0
        self.scores: list[float] = []
        self.returns: list[float] = []
        self.wins: list[int] = []
        self.zleft: list[int] = []

    def _maybe_save_first_win(self, score: float, zleft) -> None:
        if self._saved_first_win or (self.model is None):
            return
        os.makedirs(os.path.dirname(self.first_win_path), exist_ok=True)
        self.model.save(self.first_win_path)
        self._saved_first_win = True
        if self.verbose:
            print(f"[FIRST WIN] episode={self.ep} score={score:.0f} zerglings_left={zleft} -> saved {self.first_win_path}.zip")

    def _try_save_replay(self) -> None:
        if self.training_env is None:
            return
        try:
            # will call EpisodeStatsWrapper.save_replay -> env.save_replay
            self.training_env.env_method("save_replay")
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Could not save replay via env_method('save_replay'): {e}")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            extra = info.get("episode_extra", None)
            if extra is None:
                continue

            self.ep += 1
            score = float(extra.get("score_total", 0.0))
            ret = float(extra.get("ep_return", 0.0))
            win = extra.get("win", 0)
            zleft = extra.get("zerglings_left", None)

            self.scores.append(score)
            self.returns.append(ret)

            win_int: Optional[int] = None
            try:
                win_int = int(win) if win is not None else 0
            except Exception:
                win_int = 0

            self.wins.append(win_int)

            if self.save_on_first_win and win_int == 1 and not self._saved_first_win:
                self._maybe_save_first_win(score=score, zleft=zleft)

            if zleft is not None:
                try:
                    self.zleft.append(int(zleft))
                except Exception:
                    pass

            # Replay trigger from wrapper
            if info.get("save_replay", False):
                if self.verbose:
                    print(f"[REPLAY] Triggered at episode {self.ep}")
                self._try_save_replay()

            # --- rolling win rate (always) ---
            recent_wins = self.wins[-min(len(self.wins), self.print_every):]
            win_rate = float(np.mean(recent_wins)) if len(recent_wins) else 0.0

            # TensorBoard
            self.logger.record("custom/score_total", score)
            self.logger.record("custom/ep_return", ret)
            self.logger.record("custom/win", float(win_int))
            self.logger.record("custom/win_rate", win_rate)

            # extra debug tags (handy)
            self.logger.record("custom/episodes_seen", float(extra.get("episodes_seen", self.ep)))
            self.logger.record("custom/last_win", float(extra.get("last_win", win_int)))

            if zleft is not None:
                self.logger.record("custom/zerglings_left", float(zleft))

            # Console print
            if self.verbose and (self.ep % self.print_every == 0):
                mean_score = float(np.mean(self.scores[-self.print_every:]))
                mean_ret = float(np.mean(self.returns[-self.print_every:]))
                msg = f"[episodes={self.ep}] mean_score={mean_score:.2f} mean_return={mean_ret:.2f} win_rate={win_rate:.2f}"
                if len(self.zleft) >= self.print_every:
                    mean_z = float(np.mean(self.zleft[-self.print_every:]))
                    msg += f" zerglings_left={mean_z:.2f}"
                print(msg)

        return True
