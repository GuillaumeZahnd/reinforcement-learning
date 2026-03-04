import gymnasium as gym
from sb3_contrib import MaskablePPO
from minigrid.wrappers import ImgObsWrapper
from pathlib import Path

from environment_wrapper import EnvironmentWrapper
from utils import print_green, print_red


if __name__ == "__main__":

    # Puzzle to solve
    puzzle_name = "MiniGrid-DoorKey-8x8-v0"

    # Environment
    max_steps = 100  # Budget allowed to solve the puzzle
    env = gym.make(puzzle_name, max_steps=max_steps, render_mode="human")

    # Standard observation wrapper
    env = ImgObsWrapper(env)

    # Custom wrapper for assigning intermediate rewards and masking futile actions
    env = EnvironmentWrapper(env=env, print_rewards=True)

    # Agent
    agent_name = "MaskablePPO_MLP"

    # Load
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{puzzle_name}_{agent_name}"
    load_path = checkpoint_dir / file_name
    model = MaskablePPO.load(load_path)

    # Reset
    obs, info = env.reset()

    # Run
    nb_steps = 1000
    episode_len = 0
    for i in range(nb_steps):

        current_mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=current_mask, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        episode_len += 1

        if terminated:
            print_green(f"✨ Episode finished successfully in {episode_len} steps.")
            obs, _ = env.reset()
            episode_len = 0

        if truncated:
            print_red(f"Episode truncated.")
            obs, _ = env.reset()
            episode_len = 0

