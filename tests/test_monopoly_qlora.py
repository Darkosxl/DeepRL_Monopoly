from __future__ import annotations

import copy
import json
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PPO_ROOT = ROOT / "RL_PPO(UNOFFICIAL)_MONOPOLY"
SLM_ROOT = ROOT / "SLM_HANDMADE_MONOPOLY"
sys.path[:0] = [str(PPO_ROOT), str(SLM_ROOT)]

from monopoly_drl.actions import ACTION_SPACE_SIZE, OFFSETS, PROPERTY_IDS, ActionType  # noqa: E402
from monopoly_drl.agent_ppo import PPOAgent  # noqa: E402
from monopoly_drl.env import MonopolyEnv  # noqa: E402
from monopoly_qlora import (  # noqa: E402
    DecisionFormatError,
    action_to_json,
    canonical_prompt,
    deterministic_ppo_action,
    fallback_action,
    make_dataset_row,
    parse_action_json,
    play_model_game,
    serialize_decision,
    sha256_text,
    shortlist_actions,
    scripted_opponents,
    split_by_game,
    tokenize_rows,
    validate_dataset_row,
    validate_splits,
)


class PrefixTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        self.assert_tokenize = tokenize
        text = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


class MonopolyQLoRAContractsTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(4)
        self.env = MonopolyEnv(agent_ids=[0], max_rounds=5)
        self.env.turn_order = [0, 1, 2, 3]
        self.env.current_turn_idx = 0

    def give_property(self, square: int, pid: int) -> None:
        prop = self.env.properties[square]
        prop.owner = pid
        self.env.players[pid].properties.append(prop)
        self.env._update_monopolies()

    def test_every_action_id_has_an_exact_json_round_trip(self) -> None:
        for action in range(ACTION_SPACE_SIZE):
            raw = action_to_json(action, self.env, 0)
            self.assertEqual(parse_action_json(raw, self.env, 0), action)
            self.assertNotIn("```", raw)
        self.env.turn_order = [2, 3, 0, 1]
        for action in range(ACTION_SPACE_SIZE):
            raw = action_to_json(action, self.env, 2)
            self.assertEqual(parse_action_json(raw, self.env, 2), action)

    def test_every_reachable_legal_action_round_trips_with_legality_check(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 1)
        self.give_property(5, 2)
        self.give_property(15, 3)
        for action in self.env.get_allowed_actions(0):
            raw = action_to_json(action, self.env, 0)
            self.assertEqual(
                parse_action_json(raw, self.env, 0, self.env.get_allowed_actions(0)),
                action,
            )

    def test_parser_rejects_fences_extra_keys_and_illegal_actions(self) -> None:
        with self.assertRaises(DecisionFormatError):
            parse_action_json('```json\n{"action":"end_turn"}\n```', self.env, 0)
        with self.assertRaises(DecisionFormatError):
            parse_action_json('{"action":"end_turn","why":"done"}', self.env, 0)
        with self.assertRaises(DecisionFormatError):
            parse_action_json(
                '{"action":"roll_dice"}',
                self.env,
                0,
                self.env.get_allowed_actions(0),
            )

    def test_fallback_priority_is_safe(self) -> None:
        self.assertEqual(
            fallback_action([int(ActionType.ROLL_DICE), int(ActionType.END_TURN)]),
            int(ActionType.END_TURN),
        )
        self.assertEqual(fallback_action([int(ActionType.ROLL_DICE)]), int(ActionType.ROLL_DICE))
        self.assertEqual(fallback_action([987]), 987)

    def test_serialization_is_deterministic_and_seat_invariant(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 1)
        self.env.players[0].cash = 1234
        self.env.players[1].position = 17
        first = serialize_decision(self.env, 0)
        self.assertEqual(first, serialize_decision(self.env, 0))

        rotated = MonopolyEnv(agent_ids=[2], max_rounds=5)
        rotated.turn_order = [2, 3, 0, 1]
        rotated.current_turn_idx = 0
        rotated.players[2].cash = 1234
        rotated.players[3].position = 17
        for square, pid in ((1, 2), (3, 3)):
            prop = rotated.properties[square]
            prop.owner = pid
            rotated.players[pid].properties.append(prop)
        rotated._update_monopolies()

        self.assertEqual(first, serialize_decision(rotated, 2))
        self.assertEqual(canonical_prompt(self.env, 0), canonical_prompt(rotated, 2))

    def test_shortlist_keeps_teacher_mandatory_and_best_family_actions(self) -> None:
        legal = [
            int(ActionType.END_TURN),
            int(ActionType.ROLL_DICE),
            OFFSETS["mortgage"],
            OFFSETS["mortgage"] + 1,
            OFFSETS["sell_prop"],
        ]
        selected = shortlist_actions(
            legal,
            OFFSETS["mortgage"],
            {OFFSETS["mortgage"] + 1: 0.9, OFFSETS["sell_prop"]: 0.8},
            limit=4,
        )
        self.assertEqual(selected[0], OFFSETS["mortgage"])
        self.assertIn(int(ActionType.ROLL_DICE), selected)
        self.assertIn(OFFSETS["mortgage"] + 1, selected)
        self.assertLessEqual(len(selected), 4)

    def test_deterministic_hybrid_teacher_keeps_neural_candidate_scores(self) -> None:
        self.env.phase = "post_roll"
        self.env.has_rolled = True
        self.env.players[0].position = 1
        agent = PPOAgent(0, hybrid=True, device="cpu")
        action, probabilities = deterministic_ppo_action(agent, self.env, 0)
        self.assertEqual(action, int(ActionType.BUY_PROPERTY))
        self.assertIn(int(ActionType.BUY_PROPERTY), probabilities)
        self.assertIn(int(ActionType.END_TURN), probabilities)

    def test_randomized_teacher_groups_draw_from_all_six_personalities(self) -> None:
        classes = {
            type(agent)
            for seed in range(20)
            for agent in scripted_opponents(seed, teacher_pid=0)
        }
        self.assertEqual(len(classes), 6)

    def test_standalone_model_game_finishes_and_records_fallbacks(self) -> None:
        env = MonopolyEnv(agent_ids=[0], max_rounds=1)
        report = play_model_game(
            env,
            model_pid=0,
            generate=lambda _prompt: "not json",
            opponents=scripted_opponents(5, teacher_pid=0),
        )
        self.assertTrue(report["finished"])
        self.assertTrue(report["decisions"])
        self.assertTrue(all(item["fallback"] for item in report["decisions"]))

    def test_dataset_row_preserves_legal_label_and_candidate_scores(self) -> None:
        self.give_property(1, 0)
        action = OFFSETS["mortgage"] + PROPERTY_IDS.index(1)
        row = make_dataset_row(
            env=self.env,
            actor_pid=0,
            game_id="game-1",
            seed=10,
            step=3,
            teacher_action=action,
            relabeled_action=action,
            rollout_scores={action: [0.1, 0.2, 0.3, 0.4]},
            outcome="loss",
            teacher_checkpoint_hash="a" * 64,
        )
        validate_dataset_row(row)
        self.assertEqual(json.loads(row["completion"])["action"], "mortgage")
        self.assertEqual(row["state_hash"], sha256_text(row["prompt"]))

        broken = copy.deepcopy(row)
        broken["completion"] = '{"action":"end_turn"}'
        with self.assertRaisesRegex(ValueError, "relabeled action"):
            validate_dataset_row(broken)

    def test_split_is_exact_deduplicated_and_has_no_game_leakage(self) -> None:
        rows = []
        for game in range(8):
            for step in range(3):
                rows.append(
                    {
                        "game_id": f"g{game}",
                        "state_hash": f"h{game}-{step}",
                        "phase": "pre_roll" if step % 2 else "post_roll",
                        "action_family": "mortgage" if step % 2 else "end_turn",
                        "step": step,
                    }
                )
        rows.append(dict(rows[0]))
        splits = split_by_game(rows, {"train": 9, "validation": 6, "test": 3}, seed=1)
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 9, "validation": 6, "test": 3})
        game_sets = [{row["game_id"] for row in split} for split in splits.values()]
        self.assertTrue(game_sets[0].isdisjoint(game_sets[1]))
        self.assertTrue(game_sets[0].isdisjoint(game_sets[2]))
        self.assertTrue(game_sets[1].isdisjoint(game_sets[2]))
        hashes = [row["state_hash"] for split in splits.values() for row in split]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_validate_splits_rejects_game_leakage(self) -> None:
        self.give_property(1, 0)
        action = OFFSETS["mortgage"] + PROPERTY_IDS.index(1)
        row = make_dataset_row(
            env=self.env,
            actor_pid=0,
            game_id="same-game",
            seed=1,
            step=1,
            teacher_action=action,
            relabeled_action=action,
            rollout_scores={action: [1.0]},
            outcome="win",
            teacher_checkpoint_hash="b" * 64,
        )
        other = copy.deepcopy(row)
        other["prompt"] += " "
        other["state_hash"] = sha256_text(other["prompt"])
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_splits({"train": [row], "test": [other]})

    def test_tokenizer_never_truncates_and_enforces_overflow_boundary(self) -> None:
        tokenizer = PrefixTokenizer()
        row = {"prompt": "state", "completion": '{"action":"end_turn"}'}
        tokenized = tokenize_rows([row], tokenizer, max_length=512)
        self.assertEqual(len(tokenized[0]["input_ids"]), tokenized[0]["token_count"])
        response_start = next(
            index for index, label in enumerate(tokenized[0]["labels"]) if label != -100
        )
        self.assertTrue(all(label == -100 for label in tokenized[0]["labels"][:response_start]))

        short = {"prompt": "", "completion": ""}
        rows = [row] + [short] * 199
        boundary = tokenize_rows(rows, tokenizer, max_length=40)
        self.assertEqual(len(boundary), 200)
        with self.assertRaisesRegex(ValueError, "overflow gate"):
            tokenize_rows([row], tokenizer, max_length=40)


if __name__ == "__main__":
    unittest.main()
