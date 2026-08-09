from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import os
import pickle
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training_guard import GIB, MemoryLimitReached, MemoryWatchdog  # noqa: E402
from monopoly_game_engine.actions import ACTION_SPACE_SIZE  # noqa: E402
from monopoly_game_engine.constants import NUM_PLAYERS, RULESET_VERSION  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv  # noqa: E402
from monopoly_game_engine.state import STATE_DIM  # noqa: E402


CHECKPOINT_VERSION = 3


@dataclass
class SharedGame:
    """A cloneable PPO-plus game with private stdlib RNG state."""

    env: MonopolyEnv
    random_state: object

    @classmethod
    def new(cls, seed: int, max_rounds: int) -> "SharedGame":
        outer_state = random.getstate()
        try:
            random.seed(seed)
            env = MonopolyEnv(agent_ids=[0], max_rounds=max_rounds)
            game_state = random.getstate()
        finally:
            random.setstate(outer_state)
        return cls(env=env, random_state=game_state)

    def clone(self) -> "SharedGame":
        return copy.deepcopy(self)

    def reseed(self, seed: int) -> None:
        self.random_state = random.Random(seed).getstate()

    def step(self, action: int):
        outer_state = random.getstate()
        try:
            random.setstate(self.random_state)
            result = self.env.step(action)
            self.random_state = random.getstate()
            return result
        finally:
            random.setstate(outer_state)

    def legal_actions(self) -> tuple[int, ...]:
        return tuple(self.env.get_allowed_actions(self.env.whose_turn()))

    def information_key(self, player_id: int) -> bytes:
        players = tuple(
            (
                player.player_id,
                player.cash,
                player.position,
                player.in_jail,
                player.jail_turns,
                player.gooj_card,
                player.bankrupt,
            )
            for player in self.env.players
        )
        properties = tuple(
            (
                square,
                prop.owner,
                prop.mortgaged,
                prop.houses,
            )
            for square, prop in sorted(self.env.properties.items())
        )
        offers = tuple(
            sorted(
                (
                    sender,
                    offer.to_player,
                    getattr(offer.offered_prop, "square_id", None),
                    getattr(offer.requested_prop, "square_id", None),
                    offer.cash_offered,
                    offer.cash_requested,
                )
                for sender, offer in self.env.pending_trades.items()
            )
        )
        public = (
            player_id,
            players,
            properties,
            tuple(self.env.turn_order),
            self.env.current_turn_idx,
            self.env.round,
            self.env.max_rounds,
            self.env.phase,
            self.env.has_rolled,
            tuple(self.env.out_of_turn_pids),
            self.env.last_dice,
            self.env.consecutive_doubles,
            self.env.extra_roll_pending,
            self.env.auction_property_id,
            tuple(self.env.auction_bidders),
            self.env.auction_current_pid,
            self.env.auction_high_bid,
            self.env.auction_high_bidder,
            self.env.houses_available,
            self.env.hotels_available,
            self.env.debt_player,
            self.env.debt_creditor,
            self.env.debt_amount,
            self.legal_actions(),
            offers,
        )
        return hashlib.blake2b(
            pickle.dumps(public, protocol=4), digest_size=16
        ).digest()


@dataclass
class InfoSet:
    actions: tuple[int, ...]
    regret_sum: np.ndarray
    strategy_sum: np.ndarray
    visits: int = 0

    @classmethod
    def new(cls, actions: tuple[int, ...]) -> "InfoSet":
        size = len(actions)
        return cls(
            actions=actions,
            regret_sum=np.zeros(size, dtype=np.float32),
            strategy_sum=np.zeros(size, dtype=np.float32),
        )

    def current_policy(self) -> np.ndarray:
        positive = np.maximum(self.regret_sum, 0)
        total = float(positive.sum())
        if total > 0:
            return positive / total
        return np.full(len(self.actions), 1 / len(self.actions), dtype=np.float32)

    def average_policy(self) -> np.ndarray:
        total = float(self.strategy_sum.sum())
        if total > 0:
            return self.strategy_sum / total
        return self.current_policy()


class _CheckpointUnpickler(pickle.Unpickler):
    """Map version-one `python -m` checkpoints back to this module."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "InfoSet":
            return InfoSet
        return super().find_class(module, name)


@dataclass(frozen=True)
class CFRConfig:
    sims_per_action: int = 1
    rollout_horizon: int = 256
    epsilon: float = 0.1
    max_rounds: int = 200
    max_decisions: int = 20_000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sims_per_action < 1 or self.rollout_horizon < 1:
            raise ValueError("Simulation counts and rollout horizon must be positive")
        if not 0 <= self.epsilon <= 1:
            raise ValueError("epsilon must be between zero and one")


class MonteCarloCFR:
    """CFR-style rollout regret matching over the shared PPO-plus game.

    This practical approximation does not apply counterfactual reach weights
    or MCCFR sampling corrections and therefore has no equilibrium guarantee.
    The historical class name remains for checkpoint and import compatibility.
    """

    def __init__(self, config: CFRConfig | None = None):
        self.config = config or CFRConfig()
        self.tables: list[dict[bytes, InfoSet]] = [
            {} for _ in range(NUM_PLAYERS)
        ]
        self.rng = random.Random(self.config.seed)
        self.games_trained = 0
        self.in_progress: dict | None = None

    def _get_infoset(self, game: SharedGame, player_id: int) -> InfoSet:
        key = game.information_key(player_id)
        table = self.tables[player_id]
        if key not in table:
            table[key] = InfoSet.new(game.legal_actions())
        return table[key]

    @staticmethod
    def _sample(
        actions: tuple[int, ...], probabilities: np.ndarray, rng: random.Random
    ) -> int:
        return rng.choices(actions, weights=probabilities.tolist(), k=1)[0]

    def _rollout_policy(
        self, game: SharedGame, player_id: int
    ) -> tuple[tuple[int, ...], np.ndarray]:
        actions = game.legal_actions()
        node = self.tables[player_id].get(game.information_key(player_id))
        if node is None:
            return actions, np.full(
                len(actions), 1 / len(actions), dtype=np.float32
            )
        return actions, node.current_policy()

    def utility(self, game: SharedGame, player_id: int) -> float:
        player = game.env.players[player_id]
        if player.bankrupt:
            return -1.0
        if game.env.done:
            return 1.0 if game.env.winner() == player_id else -1.0

        others = [
            p.net_worth()
            for p in game.env.players
            if p.player_id != player_id and not p.bankrupt
        ]
        if not others:
            return 1.0
        own = player.net_worth()
        mean_other = float(np.mean(others))
        return float(
            np.clip(
                (own - mean_other) / (abs(own) + abs(mean_other) + 1e-9),
                -1,
                1,
            )
        )

    def _evaluate_action(
        self,
        game: SharedGame,
        action: int,
        target_player: int,
        seed: int,
    ) -> float:
        rollout = game.clone()
        rollout.reseed(seed)
        rollout.step(action)
        rollout_rng = random.Random(seed ^ 0x9E3779B9)

        for _ in range(self.config.rollout_horizon):
            if rollout.env.done:
                break
            player_id = rollout.env.whose_turn()
            actions, probabilities = self._rollout_policy(rollout, player_id)
            rollout.step(self._sample(actions, probabilities, rollout_rng))
        return self.utility(rollout, target_player)

    def optimize(
        self, game: SharedGame, game_index: int, decision_index: int
    ) -> tuple[int, dict[str, float]]:
        player_id = game.env.whose_turn()
        node = self._get_infoset(game, player_id)
        policy = node.current_policy()
        utilities = np.zeros(len(node.actions), dtype=np.float32)

        for action_index, action in enumerate(node.actions):
            samples = []
            for simulation in range(self.config.sims_per_action):
                seed = (
                    self.config.seed
                    + game_index * 1_000_003
                    + decision_index * 1_009
                    + simulation
                )
                samples.append(
                    self._evaluate_action(game, action, player_id, seed)
                )
            utilities[action_index] = float(np.mean(samples))

        expected = float(np.dot(policy, utilities))
        node.regret_sum += utilities - expected
        node.strategy_sum += policy
        node.visits += 1

        updated_policy = node.current_policy()
        if self.rng.random() < self.config.epsilon:
            selected = self.rng.choice(node.actions)
        else:
            selected = self._sample(node.actions, updated_policy, self.rng)

        return selected, {
            "player": float(player_id),
            "actions": float(len(node.actions)),
            "expected_utility": expected,
            "max_utility": float(utilities.max()),
        }

    def train_game(
        self,
        game_index: int | None = None,
        watchdog: MemoryWatchdog | None = None,
        progress_every: int = 0,
        checkpoint_every_decisions: int = 0,
        checkpoint_path: str | Path | None = None,
    ) -> dict:
        index = self.games_trained if game_index is None else game_index
        resumed = self.in_progress is not None
        if resumed:
            progress = self.in_progress
            if progress["game_index"] != index:
                raise ValueError(
                    f"Cannot start game {index}; game {progress['game_index']} "
                    "is still in progress"
                )
            game = progress["game"]
            decisions = progress["decisions"]
            elapsed_before = progress["elapsed_seconds"]
            if decisions >= self.config.max_decisions:
                raise ValueError(
                    f"Game already reached {decisions} decisions; increase "
                    "max_decisions before resuming"
                )
        else:
            game = SharedGame.new(
                self.config.seed + index, self.config.max_rounds
            )
            decisions = 0
            elapsed_before = 0.0
            self.in_progress = {
                "game_index": index,
                "game": game,
                "decisions": decisions,
                "elapsed_seconds": elapsed_before,
            }
        started = time.perf_counter()

        while not game.env.done and decisions < self.config.max_decisions:
            if watchdog is not None:
                watchdog.check()
            action, details = self.optimize(game, index, decisions)
            game.step(action)
            decisions += 1
            elapsed = elapsed_before + time.perf_counter() - started
            self.in_progress.update(
                decisions=decisions,
                elapsed_seconds=elapsed,
            )
            if progress_every > 0 and decisions % progress_every == 0:
                print(
                    {
                        "game": index,
                        "round": game.env.round,
                        "decisions": decisions,
                        "legal_actions": int(details["actions"]),
                        "infosets": sum(len(table) for table in self.tables),
                        "seconds": elapsed,
                    },
                    flush=True,
                )
            if (
                checkpoint_path is not None
                and checkpoint_every_decisions > 0
                and decisions % checkpoint_every_decisions == 0
            ):
                self.save(checkpoint_path)

        completed = game.env.done
        elapsed = elapsed_before + time.perf_counter() - started
        if completed:
            self.games_trained = max(self.games_trained, index + 1)
            self.in_progress = None
        return {
            "game": index,
            "winner": game.env.winner(),
            "rounds": game.env.round,
            "decisions": decisions,
            "truncated": not completed,
            "infosets": sum(len(table) for table in self.tables),
            "seconds": elapsed,
            "resumed": resumed,
        }

    def play_game(self, seed: int, use_average: bool = True) -> dict:
        game = SharedGame.new(seed, self.config.max_rounds)
        rng = random.Random(seed)
        decisions = 0
        while not game.env.done and decisions < self.config.max_decisions:
            player_id = game.env.whose_turn()
            actions = game.legal_actions()
            node = self.tables[player_id].get(game.information_key(player_id))
            probabilities = (
                node.average_policy()
                if node is not None and use_average
                else node.current_policy()
                if node is not None
                else np.full(len(actions), 1 / len(actions), dtype=np.float32)
            )
            game.step(self._sample(actions, probabilities, rng))
            decisions += 1
        return {
            "winner": game.env.winner(),
            "rounds": game.env.round,
            "decisions": decisions,
            "truncated": not game.env.done,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        payload = {
            "format_version": CHECKPOINT_VERSION,
            "ruleset": RULESET_VERSION,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_SPACE_SIZE,
            "config": asdict(self.config),
            "games_trained": self.games_trained,
            "tables": [
                {
                    key: (
                        node.actions,
                        node.regret_sum,
                        node.strategy_sum,
                        node.visits,
                    )
                    for key, node in table.items()
                }
                for table in self.tables
            ],
            "rng_state": self.rng.getstate(),
            "in_progress": (
                None
                if self.in_progress is None
                else {
                    "game_index": self.in_progress["game_index"],
                    "env": self.in_progress["game"].env,
                    "random_state": self.in_progress["game"].random_state,
                    "decisions": self.in_progress["decisions"],
                    "elapsed_seconds": self.in_progress["elapsed_seconds"],
                }
            ),
        }
        with gzip.open(temporary, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path) -> "MonteCarloCFR":
        with gzip.open(path, "rb") as handle:
            payload = _CheckpointUnpickler(handle).load()
        version = payload.get("format_version", 1)
        if version not in (1, 2, CHECKPOINT_VERSION):
            raise ValueError(f"Unsupported CFR checkpoint version: {version}")
        expected = (RULESET_VERSION, STATE_DIM, ACTION_SPACE_SIZE)
        actual = (
            payload.get("ruleset"),
            payload.get("state_dim"),
            payload.get("action_dim"),
        )
        if actual != expected:
            raise ValueError(f"Incompatible CFR checkpoint: {actual}, expected {expected}")
        model = cls(CFRConfig(**payload["config"]))
        model.tables = []
        for table in payload["tables"]:
            restored = {}
            for key, value in table.items():
                if isinstance(value, InfoSet):
                    restored[key] = value
                    continue
                actions, regret_sum, strategy_sum, visits = value
                restored[key] = InfoSet(
                    actions=tuple(actions),
                    regret_sum=np.asarray(regret_sum, dtype=np.float32),
                    strategy_sum=np.asarray(strategy_sum, dtype=np.float32),
                    visits=int(visits),
                )
            model.tables.append(restored)
        model.games_trained = payload["games_trained"]
        model.rng.setstate(payload["rng_state"])
        progress = payload.get("in_progress")
        if progress is not None:
            model.in_progress = {
                "game_index": progress["game_index"],
                "game": SharedGame(
                    env=progress["env"],
                    random_state=progress["random_state"],
                ),
                "decisions": progress["decisions"],
                "elapsed_seconds": progress["elapsed_seconds"],
            }
        return model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the PPO-plus CFR-style rollout regret matcher"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--games", type=int, default=1)
    train.add_argument("--sims-per-action", type=int, default=1)
    train.add_argument("--rollout-horizon", type=int, default=256)
    train.add_argument("--max-rounds", type=int, default=200)
    train.add_argument("--max-decisions", type=int)
    train.add_argument("--epsilon", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument(
        "--checkpoint",
        default=str(ROOT / "artifacts" / "cfr_ppo_plus" / "cfr.pkl.gz"),
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument("--stop-rss-gib", type=float, default=3)
    train.add_argument("--hard-rss-gib", type=float, default=4)
    train.add_argument("--min-available-gib", type=float, default=2)
    train.add_argument("--progress-every", type=int, default=10)
    train.add_argument("--checkpoint-every-decisions", type=int, default=100)

    play = subparsers.add_parser("play")
    play.add_argument("checkpoint")
    play.add_argument("--games", type=int, default=10)
    play.add_argument("--seed", type=int, default=10_000)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "play":
        model = MonteCarloCFR.load(args.checkpoint)
        for game in range(args.games):
            print(model.play_game(args.seed + game))
        return

    checkpoint = Path(args.checkpoint)
    if args.resume and checkpoint.exists():
        model = MonteCarloCFR.load(checkpoint)
        if args.max_decisions is not None:
            model.config = replace(
                model.config,
                max_decisions=args.max_decisions,
            )
    else:
        model = MonteCarloCFR(
            CFRConfig(
                sims_per_action=args.sims_per_action,
                rollout_horizon=args.rollout_horizon,
                epsilon=args.epsilon,
                max_rounds=args.max_rounds,
                max_decisions=args.max_decisions or 20_000,
                seed=args.seed,
            )
        )

    watchdog = MemoryWatchdog(
        stop_rss_gib=args.stop_rss_gib,
        hard_rss_gib=args.hard_rss_gib,
        min_available_gib=args.min_available_gib,
    )
    for _ in range(args.games):
        try:
            stats = model.train_game(
                watchdog=watchdog,
                progress_every=args.progress_every,
                checkpoint_every_decisions=args.checkpoint_every_decisions,
                checkpoint_path=checkpoint,
            )
        except MemoryLimitReached as exc:
            emergency = checkpoint.with_name(
                f"{checkpoint.stem}_emergency{checkpoint.suffix}"
            )
            model.save(emergency)
            print(f"Memory watchdog stopped training: {exc}")
            print(f"Emergency checkpoint: {emergency}")
            return
        except KeyboardInterrupt:
            model.save(checkpoint)
            print(f"Interrupted; partial regret tables saved to: {checkpoint}")
            return
        model.save(checkpoint)
        stats["peak_rss_gib"] = watchdog.peak_rss / GIB
        print(stats, flush=True)


if __name__ == "__main__":
    main()
