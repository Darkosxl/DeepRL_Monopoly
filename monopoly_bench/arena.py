"""Deterministic game execution and seat-balanced statistics."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping

import numpy as np

from .adapters import ActionDecision
from .contracts import GameResult, MatchSummary, ReplayPosition, SearchResult
from .engine import MAX_DECISIONS_PER_TURN, NUM_PLAYERS, SharedGame, legal_mask


def play_game(
    *,
    game_id: int,
    seed: int,
    policies: Mapping[int, object],
    max_rounds: int = 200,
    record_seats: set[int] | None = None,
) -> GameResult:
    """Run one game; illegal actions fail closed and are never substituted."""
    game = SharedGame.new(seed, max_rounds)
    result = GameResult(game_id, seed, None, False, 0)
    max_decisions = max_rounds * NUM_PLAYERS * MAX_DECISIONS_PER_TURN
    try:
        for step in range(max_decisions):
            if game.env.done:
                break
            player_id = game.env.whose_turn()
            legal = tuple(game.env.get_allowed_actions(player_id))
            if len(legal) == 1:
                action = legal[0]
            else:
                policy = policies[player_id]
                decision_seed = seed * 1_000_003 + step * 17 + player_id
                decision = policy.choose_action(game, player_id, decision_seed)
                if isinstance(decision, SearchResult):
                    action = decision.chosen_action
                    result.search_latencies.append(decision.latency_s)
                    if record_seats is None or player_id in record_seats:
                        result.positions.append(
                            ReplayPosition(
                                state=game.env._get_state(player_id),
                                legal_mask=legal_mask(legal),
                                visits=decision.visits,
                                q_values=decision.q_values,
                                selected_action=action,
                                value=decision.root_value,
                                actor_id=player_id,
                                game_id=game_id,
                            )
                        )
                elif isinstance(decision, ActionDecision):
                    action = decision.action
                    result.fallbacks += int(decision.fallback)
                else:
                    action = int(decision)
                if action not in legal:
                    result.illegal_actions += 1
                    raise RuntimeError(f"Policy for seat {player_id} chose illegal action {action}")
            game.step(action)
            result.decisions = step + 1
    except Exception as exc:
        result.crashes += 1
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    result.completed = game.env.done
    result.winner = game.env.winner() if result.completed else None
    if result.winner is not None:
        outcome = [0.0] * NUM_PLAYERS
        outcome[result.winner] = 1.0
        for position in result.positions:
            position.outcome = tuple(outcome)  # type: ignore[assignment]
    return result


def balanced_team_seats(games: int) -> list[frozenset[int]]:
    seats = [frozenset(pair) for pair in combinations(range(NUM_PLAYERS), 2)]
    return [seats[index % len(seats)] for index in range(games)]


def balanced_single_seats(games: int) -> list[frozenset[int]]:
    return [frozenset({index % NUM_PLAYERS}) for index in range(games)]


def wilson_lower(wins: int, games: int, z: float = 1.959963984540054) -> float:
    if games <= 0:
        return 0.0
    proportion = wins / games
    denominator = 1 + z * z / games
    centre = proportion + z * z / (2 * games)
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * games)) / games)
    return (centre - margin) / denominator


def summarize(results: list[GameResult], target_seats: list[frozenset[int]]) -> MatchSummary:
    if len(results) != len(target_seats):
        raise ValueError("Every result needs a target-side seat assignment")
    completed = sum(result.completed for result in results)
    wins = sum(result.completed and result.winner in seats for result, seats in zip(results, target_seats))
    losses = completed - wins
    latencies = [latency for result in results for latency in result.search_latencies]
    return MatchSummary(
        games=len(results),
        completed=completed,
        wins=wins,
        losses=losses,
        win_rate=wins / len(results) if results else 0.0,
        wilson_lower=wilson_lower(wins, len(results)),
        illegal_actions=sum(result.illegal_actions for result in results),
        fallbacks=sum(result.fallbacks for result in results),
        crashes=sum(result.crashes for result in results),
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else 0.0,
    )
