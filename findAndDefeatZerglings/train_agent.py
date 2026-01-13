from __future__ import annotations

import os
import torch
from absl import flags

FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(["train_agent.py"])

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from sc2_gym import FindAndDefeatZerglingsGym
from episode_stats_wrapper import EpisodeStatsWrapper
from callbacks import PrintEpisodeExtrasCallback


def make_env():
    env = FindAndDefeatZerglingsGym(
        map_name="FindAndDefeatZerglings",
        grid_n=8,
        step_mul=8,
        visualize=False,
        sc2_replay_dir=None,          
        replay_out_dir="sc2_replays",
    )
    env = EpisodeStatsWrapper(env, replay_every_ep=10, max_replays=10)
    return env


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("sc2_replays", exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecMonitor(env)

    callback = PrintEpisodeExtrasCallback(
        print_every=20,
        save_on_first_win=True,
        first_win_path="checkpoints/ppo_first_win",
        verbose=1,
    )

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        policy_kwargs=dict(normalize_images=False),
        n_steps=1024,
        batch_size=256,
        learning_rate=2.5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        n_epochs=10,
        device="cuda",
        tensorboard_log="./tb_logs",
    )

    model.learn(total_timesteps=1_000_000, callback=callback)
    model.save("checkpoints/ppo_final_find_and_defeat_zerglings")
    env.close()
