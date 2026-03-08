import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.core.world_object import Door, Goal, Door, Wall, Key
import numpy as np

from utils import action_name2idx, print_green


class EnvironmentWrapper(gym.Wrapper):
    """Environment wrapper for reward shaping and action masking."""
    def __init__(self, env: gym.Env, print_rewards: bool) -> None:
        super().__init__(env)

        # Intermediate rewards tracking (awarded once, on the first occurence of the event)
        self.key_pickedup = False
        self.door_opened = False
        self.door_reached = False
        self.intermediate_reward = 0.2
        self.time_penalty_per_step = 1e-3
        self.print_rewards = print_rewards

    def step(self, action):
        """
        Execute a single step in the environment and apply reward shaping.

        Args:
            action: Action index selected by the agent.

        Returns:
            Tuple containing (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        reward += self.leave_breadcrumbs(obs=obs)
        reward -= self.time_penalty_per_step
        return obs, reward, terminated, truncated, info


    def reset(self, seed: int=None, options: dict[str, any]=None) -> tuple[np.ndarray, dict[str, any]]:
        """
        Reset the environment and the flags for reward tracking.

        Returns:
            Initial observation.
            Information dictionary.
        """
        self.key_pickedup = False
        self.door_opened = False
        self.door_reached = False
        self.door_sequence = np.zeros(3)
        obs, info = super().reset(seed=seed, options=options)
        return obs, info


    def leave_breadcrumbs(self, obs: np.ndarray) -> float:
        """
        Logic for awarding intermediate rewards ("bread crumbs") based on agent progression.

        Args:
            obs: Current observation array from the environment.

        Returns:
            Total breadcrumb reward value for the current step.
        """

        breadcrumbs = 0.0

        grid = self.env.unwrapped.grid
        front_pos = self.env.unwrapped.front_pos
        tile_in_front = grid.get(*front_pos)

        unwrapped = self.env.unwrapped
        agent_pos = unwrapped.agent_pos
        tile_under_agent = unwrapped.grid.get(*agent_pos)

        # Reward for picking up the key
        if not self.key_pickedup:
            carrying = self.env.unwrapped.carrying
            if isinstance(carrying, Key):
                breadcrumbs += self.intermediate_reward
                self.key_pickedup = True
                if self.print_rewards:
                    print_green("[REWARD] The agent picked up the key.")

        # Reward for opening the door
        if not self.door_opened:
            if isinstance(tile_in_front, Door) and tile_in_front.is_open:
                breadcrumbs += self.intermediate_reward
                self.door_opened = True
                if self.print_rewards:
                    print_green("[REWARD] The agent opened the door.")

        # Reward for reaching the door
        if not self.door_reached:
            if isinstance(tile_under_agent, Door):
                breadcrumbs += self.intermediate_reward
                self.door_reached = True
                if self.print_rewards:
                    print_green("[REWARD] The agent reached the door.")

        return breadcrumbs


    def action_masks(self) -> np.ndarray:
        """
        Generate a boolean mask for the current state to prevent the agent from taking futile actions.
        This method is used by the MaskablePPO algorithm.

        Returns:
            Boolean array, of shape (nb_actions).
        """

        # Boolean mask to authorize (1) or prevent (0) actions
        # Possible actions are: 0 (left), 1 (right), 2 (forward), 3 (pickup), 4 (drop), 5 (toggle), 6 (done)
        nb_actions = len(Actions)
        mask = np.ones(nb_actions, dtype=bool)

        # Retrieve the tile located in front of the agent
        grid = self.env.unwrapped.grid
        front_pos = self.env.unwrapped.front_pos
        tile_in_front = grid.get(*front_pos)

        # Flag to indicate whether the agent is carrying anything
        carrying = self.env.unwrapped.carrying

        # Penalty for Pickup
        if not isinstance(tile_in_front, Key):
            mask[action_name2idx(name="pickup")] = False

        # Penalty for Toggle
        mask[action_name2idx(name="toggle")] = False
        if (
            isinstance(tile_in_front, Door) and
            tile_in_front.is_locked and
            isinstance(carrying, Key)
        ):
            mask[action_name2idx(name="toggle")] = True

        # Penalty for Drop
        mask[action_name2idx(name="drop")] = False

        # Penalty for Forward
        mask[action_name2idx(name="forward")] = False
        if (
            tile_in_front is None or
            (isinstance(tile_in_front, Door) and tile_in_front.is_open) or
            isinstance(tile_in_front, Goal)
        ):
            mask[action_name2idx(name="forward")] = True
        return mask

