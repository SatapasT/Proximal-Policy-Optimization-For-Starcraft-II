import os
import sys
from pathlib import Path
from datetime import datetime

import torch
from absl import flags

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecNormalize
from stable_baselines3.common.logger import configure

from .sc2_gym import DefeatRoachesGym
from common.metrics import EpisodeStatsWrapper, PrintEpisodeExtrasCallback

from absl import flags as absl_flags


def ensure_absl_flags_parsed_for_subproc() -> None:
    f = absl_flags.FLAGS
    if not f.is_parsed():
        f(["subproc"], known_only=True)


FLAGS = flags.FLAGS
flags.DEFINE_string("load_path", None, "Optional path to a saved SB3 PPO model (.zip).")
flags.DEFINE_integer("chunk_timesteps", 200_000, "Timesteps per training chunk.")
flags.DEFINE_integer("n_envs", 8, "Parallel envs.")
flags.DEFINE_integer("n_steps", 256, "Per-env rollout steps.")
flags.DEFINE_boolean("use_vecnorm_reward", True, "Use VecNormalize for reward normalization.")
flags.DEFINE_integer("seed", 0, "Random seed.")

THIS_DIR = Path(__file__).resolve().parent
CHECKPOINT_ROOT = (THIS_DIR / "checkpoints").resolve()
SAVE_EVERY_EPISODES = 500


def is_windows() -> bool:
    return sys.platform.startswith("win")


def linear_schedule(initial_value: float):
    initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def build_action_names() -> list[str]:
    tmp_env = DefeatRoachesGym(
        map_name="DefeatRoaches",
        grid_n=8,
        step_mul=8,
        visualize=False,
    )
    names = list(tmp_env.action_type_names)
    tmp_env.close()
    return names


def resolve_run_dir(load_path: str | None) -> Path:
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    if load_path and str(load_path).strip():
        p = Path(load_path).expanduser()
        p = (Path.cwd() / p).resolve() if not p.is_absolute() else p.resolve()
        run_dir = p.parent
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    run_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = (CHECKPOINT_ROOT / f"run_{run_stamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def make_env_fn(rank: int, seed: int, run_dir: Path):
    def _init():
        ensure_absl_flags_parsed_for_subproc()

        worker_tmp = (run_dir / "sc2_tmp" / f"rank_{rank}").resolve()
        worker_tmp.mkdir(parents=True, exist_ok=True)

        os.environ["TMP"] = str(worker_tmp)
        os.environ["TEMP"] = str(worker_tmp)
        os.environ["TMPDIR"] = str(worker_tmp)

        env = DefeatRoachesGym(
            map_name="DefeatRoaches",
            grid_n=8,
            step_mul=8,
            visualize=False,
        )

        env.reset(seed=seed + rank)
        env = EpisodeStatsWrapper(
            env,
            num_action_types=len(env.action_type_names),
            kill_keys=[],
        )
        return env

    return _init


if __name__ == "__main__":
    FLAGS(sys.argv)

    N_ENVS = int(FLAGS.n_envs)
    N_STEPS = int(FLAGS.n_steps)
    CHUNK_TIMESTEPS = int(FLAGS.chunk_timesteps)
    SEED = int(FLAGS.seed)
    use_vecnorm_reward = bool(FLAGS.use_vecnorm_reward)

    load_path = FLAGS.load_path
    load_path = str(load_path) if load_path and str(load_path).strip() else None

    RUN_DIR = resolve_run_dir(load_path)
    TB_DIR = (RUN_DIR / "tb").resolve()
    TB_DIR.mkdir(parents=True, exist_ok=True)

    print("run_dir:", str(RUN_DIR))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    env_fns = [make_env_fn(i, SEED, RUN_DIR) for i in range(N_ENVS)]
    if N_ENVS == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn" if is_windows() else None)

    env = VecMonitor(env)

    vecnorm_path = RUN_DIR / "vecnormalize.pkl"
    if use_vecnorm_reward:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    action_names = build_action_names()

    callback = PrintEpisodeExtrasCallback(
        action_type_names=action_names,
        print_every=20,
        log_dir=RUN_DIR,
        save_every_episodes=SAVE_EVERY_EPISODES,
        save_dir=RUN_DIR,
        verbose=1,
        print_summary=True,
        write_summary_json=True,
        log_episode_json_every=1,
        win_rate_window=50, 
    )


    rollout_size = N_STEPS * N_ENVS
    batch_size = 256
    if rollout_size % batch_size != 0:
        print(f"[WARN] rollout_size={rollout_size} not divisible by batch_size={batch_size}.")

    lr = linear_schedule(2.5e-4)

    if load_path:
        print(f"[RESUME/TRANSFER] Loading model from: {load_path}")
        model = PPO.load(load_path, env=env, device="cuda", print_system_info=True)

        if use_vecnorm_reward and vecnorm_path.exists():
            env = VecNormalize.load(str(vecnorm_path), env)
            env.training = True
            env.norm_reward = True
            model.set_env(env)
    else:
        print("[START] Training from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            policy_kwargs=dict(normalize_images=False),
            n_steps=N_STEPS,
            batch_size=batch_size,
            learning_rate=lr,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_epochs=5,
            ent_coef=0.02,
            target_kl=0.03,
            max_grad_norm=0.5,
            vf_coef=0.5,
            device="cuda",
            tensorboard_log=str(TB_DIR),
            seed=SEED,
        )

    sb3_logger = configure(folder=str(RUN_DIR), format_strings=["stdout", "log", "tensorboard"])
    model.set_logger(sb3_logger)

    try:
        while True:
            model.learn(
                total_timesteps=CHUNK_TIMESTEPS,
                callback=callback,
                reset_num_timesteps=False,
            )
            if use_vecnorm_reward:
                env.save(str(vecnorm_path))
    except KeyboardInterrupt:
        ep = getattr(callback, "ep", None)
        interrupt_path = RUN_DIR / (f"ppo_interrupt_ep{ep:06d}" if isinstance(ep, int) else "ppo_interrupt")
        print("\n[INTERRUPT] Saving model...")
        model.save(str(interrupt_path))
        print(f"[INTERRUPT] Saved -> {interrupt_path}.zip")
        if use_vecnorm_reward:
            env.save(str(vecnorm_path))
    finally:
        try:
            env.close()
        except (EOFError, BrokenPipeError):
            pass
