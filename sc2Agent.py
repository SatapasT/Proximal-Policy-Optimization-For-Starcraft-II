# sc2Agent.py
import numpy as np

from pysc2.env import sc2_env, run_loop
from pysc2.agents import base_agent
from pysc2.lib import actions, features

from absl import app



# Make numpy prints a bit nicer in the console
np.set_printoptions(
    linewidth=120,
    suppress=True,
    threshold=50  # don't print massive arrays fully
)


class DebugAgent(base_agent.BaseAgent):
    """Agent that just prints out stuff about the observation and does no-op."""

    def step(self, timestep):
        # Keep BaseAgent bookkeeping (steps, reward, episodes, etc.)
        super().step(timestep)

        obs = timestep.observation

        # Only spam a little: first episode, first few steps
        if self.episodes == 1 and self.steps <= 5:
            print("\n" + "=" * 80)
            print(f"EPISODE {self.episodes}  STEP {self.steps}")
            print("=" * 80)

            self.print_basic_info(timestep)
            self.print_minimap_and_screen(obs)
            self.print_available_actions(obs)
            self.print_units_info(obs)

        # Agent does nothing – pure no-op
        return actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])

    # ----------------- helper print functions -----------------

    def print_basic_info(self, timestep):
        print("\n[Basic timestep info]")
        print(f"reward this step : {timestep.reward}")
        print(f"discount        : {timestep.discount}")

        o = timestep.observation
        print("\nObservation keys:", list(o.keys()))

        if "player" in o:
            print("\n[player]")
            print(o["player"])  # minerals, vespene, supply, etc.

        if "score_cumulative" in o:
            print("\n[score_cumulative]")
            print(o["score_cumulative"])

    def print_minimap_and_screen(self, o):
        print("\n[Feature layers]")

        if "feature_minimap" in o:
            minimap = o["feature_minimap"]
            print(f"minimap shape: {minimap.shape}, dtype={minimap.dtype}")

        if "feature_screen" in o:
            screen = o["feature_screen"]
            print(f"screen  shape: {screen.shape}, dtype={screen.dtype}")

        # Show what each minimap / screen layer represents
        print("\n[Minimap feature definitions]")
        for i, f in enumerate(features.MINIMAP_FEATURES):
            print(f"  index {i:2d}: {f.name:20s} (type={f.type})")

        print("\n[Screen feature definitions]")
        for i, f in enumerate(features.SCREEN_FEATURES):
            print(f"  index {i:2d}: {f.name:20s} (type={f.type})")

    def print_available_actions(self, o):
        print("\n[Available actions this step]")

        avail = o.get("available_actions", [])
        print("IDs:", avail)

        print("\n[Decoded action names + args]")
        for a_id in avail[:20]:  # don’t spam everything
            fn = actions.FUNCTIONS[a_id]
            arg_desc = ", ".join([f"{a.name}:{a.sizes}" for a in fn.args])
            print(f"  {a_id:3d}: {fn.name:30s} ({arg_desc})")

    def print_units_info(self, o):
        print("\n[Feature / raw units]")

        f_units = o.get("feature_units", None)
        if f_units is not None and len(f_units) > 0:
            print(f"feature_units count: {len(f_units)}")
            print("first feature_unit example:")
            print(f_units[0])

        r_units = o.get("raw_units", None)
        if r_units is not None and len(r_units) > 0:
            print(f"\nraw_units count: {len(r_units)}")
            print("first raw_unit example:")
            print(r_units[0])


def main(argv):
    # argv is required by absl.app; we don't need it.
    del argv

    # Simple “minimap playground” on a DeepMind minigame
    agent = DebugAgent()

    interface_format = features.AgentInterfaceFormat(
        feature_dimensions=features.Dimensions(screen=(84, 84), minimap=(64, 64)),
        use_feature_units=True,
        use_raw_units=True,
    )

    with sc2_env.SC2Env(
        map_name="MoveToBeacon",
        players=[sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=interface_format,
        step_mul=8,
        game_steps_per_episode=0,
        visualize=True,
    ) as env:
        run_loop.run_loop([agent], env, max_episodes=1)


if __name__ == "__main__":
    app.run(main)

