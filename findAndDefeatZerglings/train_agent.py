"""Training entrypoint for PPO on the FindAndDefeatZerglings mini-game.

Uses SB3 PPO with parallel environments (SubprocVecEnv). Supports resuming from a
saved model by reusing the same run directory.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from absl import flags

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.logger import configure

from sc2_gym import FindAndDefeatZerglingsGym
from episode_stats_wrapper import EpisodeStatsWrapper
from callbacks import PrintEpisodeExtrasCallback

from absl import flags as absl_flags


def ensure_absl_flags_parsed_for_subproc() -> None:
    """Ensure absl flags are parsed inside subprocesses (Windows spawn)."""
    f = absl_flags.FLAGS
    if not f.is_parsed():
        f(["subproc"], known_only=True)


# CLI flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "load_path",
    None,
    "Optional path to a saved SB3 PPO model (.zip). If provided, training resumes from it.",
)
flags.DEFINE_integer(
    "chunk_timesteps",
    200_000,
    "Timesteps per training chunk. Training repeats chunks forever until Ctrl+C.",
)
flags.DEFINE_integer("n_envs", 4, "Number of parallel environments.")
flags.DEFINE_integer(
    "n_steps",
    512,
    "Per-env rollout steps (total rollout per update = n_steps * n_envs).",
)

THIS_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = (THIS_DIR / "checkpoints").resolve()

SAVE_EVERY_EPISODES = 500  # episode checkpoint cadence


def make_env_fn(rank: int):
    def _init():
        ensure_absl_flags_parsed_for_subproc()

        env = FindAndDefeatZerglingsGym(
            map_name="FindAndDefeatZerglings",
            grid_n=8,
            step_mul=8,
            visualize=False,
        )
        env = EpisodeStatsWrapper(env, num_action_types=len(env.action_type_names))
        return env

    return _init


def build_action_names() -> list[str]:
    """Create a temp env just to read `action_type_names`."""
    tmp_env = FindAndDefeatZerglingsGym(
        map_name="FindAndDefeatZerglings",
        grid_n=8,
        step_mul=8,
        visualize=False,
    )
    names = list(tmp_env.action_type_names)
    tmp_env.close()
    return names


def resolve_run_dir(load_path: str | None) -> Path:
    """
    If resuming, reuse the checkpoint's parent folder as RUN_DIR.
    Otherwise create a new run folder.
    """
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    if load_path and str(load_path).strip():
        p = Path(load_path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()

        run_dir = p.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (CHECKPOINT_ROOT / f"run_{run_stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def is_windows() -> bool:
    return sys.platform.startswith("win")


if __name__ == "__main__":
    # Parse CLI flags
    FLAGS(sys.argv)

    N_ENVS = int(FLAGS.n_envs)
    N_STEPS = int(FLAGS.n_steps)
    CHUNK_TIMESTEPS = int(FLAGS.chunk_timesteps)

    load_path = FLAGS.load_path
    if load_path is not None and str(load_path).strip() != "":
        load_path = str(load_path)
    else:
        load_path = None

    RUN_DIR = resolve_run_dir(load_path)

    # Run directory (checkpoints + logs)
    TB_DIR = (RUN_DIR / "tb").resolve()
    TB_DIR.mkdir(parents=True, exist_ok=True)

    print("run_dir:", str(RUN_DIR))

    # Vectorized envs:
    # - For N_ENVS == 1, avoid subprocesses entirely (prevents WinError 109 / EOFError on Ctrl+C).
    # - For N_ENVS > 1, use SubprocVecEnv (spawn on Windows).
    env_fns = [make_env_fn(i) for i in range(N_ENVS)]
    if N_ENVS == 1:
        env = DummyVecEnv(env_fns)
    else:
        start_method = "spawn" if is_windows() else None
        env = SubprocVecEnv(env_fns, start_method=start_method)

    env = VecMonitor(env)

    action_names = build_action_names()

    callback = PrintEpisodeExtrasCallback(
        action_type_names=action_names,
        print_every=20,
        win_rate_window=10,
        log_dir=RUN_DIR,
        save_every_episodes=SAVE_EVERY_EPISODES,
        save_dir=RUN_DIR,
        verbose=1,
    )

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    # PPO rollout buffer size = n_steps * n_envs
    rollout_size = N_STEPS * N_ENVS
    batch_size = 256
    if rollout_size % batch_size != 0:
        print(
            f"[WARN] batch_size={batch_size} does not divide rollout_size={rollout_size}. "
            f"Consider batch_size in {{64,128,256,512}} that divides {rollout_size}."
        )

    # Model init / resume
    if load_path:
        print(f"[RESUME] Loading model from: {load_path}")
        model = PPO.load(
            load_path,
            env=env,
            device="cuda",
            print_system_info=True,
        )
    else:
        print("[START] No --load_path provided. Training from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            policy_kwargs=dict(normalize_images=False),
            n_steps=N_STEPS,
            batch_size=batch_size,
            learning_rate=2.5e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=10,
            device="cuda",
            tensorboard_log=str(TB_DIR),
        )

    # SB3 logger sinks (console + file + TensorBoard)
    sb3_logger = configure(folder=str(RUN_DIR), format_strings=["stdout", "log", "tensorboard"])
    model.set_logger(sb3_logger)

    # Train forever in chunks
    try:
        while True:
            model.learn(
                total_timesteps=CHUNK_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=False,
            )
    except KeyboardInterrupt:
        ep = getattr(callback, "ep", None)
        if isinstance(ep, int):
            interrupt_path = RUN_DIR / f"ppo_interrupt_ep{ep:06d}"
        else:
            interrupt_path = RUN_DIR / "ppo_interrupt"
        print("\n[INTERRUPT] Caught Ctrl+C. Saving interrupt model before exit...")
        model.save(str(interrupt_path))
        print(f"[INTERRUPT] Saved -> {interrupt_path}.zip")
    finally:
        # On Windows, SubprocVecEnv can throw EOFError/BrokenPipeError during Ctrl+C teardown.
        try:
            env.close()
        except (EOFError, BrokenPipeError):
            pass
