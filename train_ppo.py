import gymnasium as gym
import minigrid
from minigrid.wrappers import ImgObsWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import MaskablePPO
from pathlib import Path

from environment_wrapper import EnvironmentWrapper

if __name__ == "__main__":

    # Puzzle to solve
    puzzle_name = "MiniGrid-DoorKey-8x8-v0"

    # Environment
    env = gym.make(puzzle_name, render_mode="rgb_array")

    # Custom wrapper for assigning intermediate rewards and masking futile actions
    env = EnvironmentWrapper(env=env, print_rewards=False)

    # Standard observation wrapper
    env = ImgObsWrapper(env)

    # Standard monitoring wrapper
    env = Monitor(env)

    # Agent
    agent_name = "MaskablePPO_MLP"
    model = MaskablePPO(
        policy="MlpPolicy",    # Policy (either CnnPolicy or MlpPolicy)
        env=env,               # Gymnasium-compatible environment
        learning_rate=0.0003,  # Learning rate
        n_steps=4096,          # Total number of steps per episode, before the environment resets
        batch_size=256,        # Number of samples per batch (nb_batches = n_steps / batch_size)
        n_epochs=10,           # Number of iterations during the optimization process
        gamma=0.99,            # Discount factor (a reward r obtained at time-step t has discounted value r*gamma^t)
        ent_coef=0.02,         # Entropy coefficient (higher values encourage more exploration)
        device="cpu",          # For small problems, the bottleneck is the transfer time, not the computation time
        verbose=1              # Print training metrics to the console
    )

    # Train
    model.learn(total_timesteps=2e6)

    # Save
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{puzzle_name}_{agent_name}"
    save_path = checkpoint_dir / file_name
    model.save(save_path)

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, deterministic=True)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

