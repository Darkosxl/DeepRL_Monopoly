"""
monopoly_drl – Hybrid Deep Reinforcement Learning for Monopoly
==============================================================

Based on:
  "Decision Making in Monopoly Using a Hybrid Deep Reinforcement
   Learning Approach"
  Bonjour et al., IEEE TETCI, Vol. 6, No. 6, December 2022.

Quick start
-----------
>>> from monopoly_drl import train_ppo, train_ddqn, evaluate_agent
>>> agent, history = train_ppo(hybrid=True, n_games=2000)
>>> results = evaluate_agent(agent, is_ppo=True, n_games=2000)
"""

from .env          import MonopolyEnv
from .agent_ppo    import PPOAgent
from .agent_ddqn   import DDQNAgent
from .agents_fixed import FPAgentA, FPAgentB, FPAgentC
from .train        import train, evaluate
from .state        import build_state_vector
from .actions      import ACTION_SPACE_SIZE, action_to_description


def train_ppo(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 2000,
    log_every: int = 100,
    **kwargs,
):
    """Train a PPO agent. Set hybrid=True for the hybrid approach."""
    agent = PPOAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    history = train(agent, is_ppo=True, hybrid=hybrid, n_games=n_games, log_every=log_every)
    return agent, history


def train_ddqn(
    hybrid: bool = True,
    player_id: int = 0,
    n_games: int = 10_000,
    log_every: int = 100,
    **kwargs,
):
    """Train a DDQN agent. Set hybrid=True for the hybrid approach."""
    agent = DDQNAgent(player_id=player_id, hybrid=hybrid, **kwargs)
    history = train(agent, is_ppo=False, hybrid=hybrid, n_games=n_games, log_every=log_every)
    return agent, history


def evaluate_agent(agent, is_ppo: bool, n_games: int = 2000, n_runs: int = 5):
    """Evaluate a trained agent against fixed-policy opponents."""
    return evaluate(agent, is_ppo=is_ppo, n_games=n_games, n_runs=n_runs)


__all__ = [
    "MonopolyEnv",
    "PPOAgent", "DDQNAgent",
    "FPAgentA", "FPAgentB", "FPAgentC",
    "train_ppo", "train_ddqn", "evaluate_agent",
    "build_state_vector", "ACTION_SPACE_SIZE", "action_to_description",
]
