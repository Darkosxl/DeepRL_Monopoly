from __future__ import annotations

import copy
import json
import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SLM_ROOT = ROOT / "SLM_HANDMADE_MONOPOLY"
sys.path.insert(0, str(SLM_ROOT))

from monopoly_game_engine.actions import ACTION_SPACE_SIZE, OFFSETS, PROPERTY_IDS, ActionType  # noqa: E402
from monopoly_game_engine.env import MonopolyEnv, TradeOffer  # noqa: E402
from monopoly_qlora import (  # noqa: E402
    SCHEMA_VERSION,
    SYSTEM_PROMPT,
    DecisionFormatError,
    action_to_json,
    asu_teacher_decision,
    asu_teacher_hash,
    canonical_json,
    canonical_prompt,
    canonical_state,
    collect_teacher_game,
    compact_dataset_prompt,
    exploratory_behavior_action,
    fallback_action,
    grouped_legal_actions,
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
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        self.assert_tokenize = tokenize
        self.assert_return_dict = return_dict
        text = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return list(text.encode("utf-8"))


class GemmaStyleTokenizer(PrefixTokenizer):
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )
        if add_generation_prompt:
            tokens.extend(b"<thought></thought>")
        return tokens


class EarlyDivergenceTokenizer(PrefixTokenizer):
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, return_dict
    ):
        tokens = super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
        )
        if add_generation_prompt:
            tokens[1] += 1
        return tokens


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

    def test_verbose_prompt_migration_matches_current_serializer(self) -> None:
        self.give_property(1, 0)
        self.give_property(3, 1)
        payload = canonical_state(self.env, 0)
        payload["legal"] = grouped_legal_actions(self.env, 0)
        old_prompt = f"old header\n{canonical_json(payload)}"
        migrated = compact_dataset_prompt(old_prompt)
        self.assertEqual(migrated, f"{SYSTEM_PROMPT}\n{serialize_decision(self.env, 0)}")

    def test_multiple_incoming_trades_are_seat_invariant(self) -> None:
        self.env.pending_trades[1] = TradeOffer(1, 0, cash_offered=100)
        self.env.pending_trades[3] = TradeOffer(3, 0, cash_offered=300)

        rotated = MonopolyEnv(agent_ids=[2], max_rounds=5)
        rotated.turn_order = [2, 3, 0, 1]
        rotated.current_turn_idx = 0
        rotated.pending_trades[3] = TradeOffer(3, 2, cash_offered=100)
        rotated.pending_trades[1] = TradeOffer(1, 2, cash_offered=300)

        self.assertEqual(serialize_decision(self.env, 0), serialize_decision(rotated, 2))

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
            {
                int(ActionType.END_TURN): 0.0,
                int(ActionType.ROLL_DICE): 0.0,
                OFFSETS["mortgage"]: 0.5,
                OFFSETS["mortgage"] + 1: 0.9,
                OFFSETS["sell_prop"]: 0.8,
            },
            legal,
            [int(ActionType.ROLL_DICE)],
            limit=4,
        )
        self.assertEqual(selected[0], OFFSETS["mortgage"])
        self.assertIn(int(ActionType.ROLL_DICE), selected)
        self.assertIn(OFFSETS["mortgage"] + 1, selected)
        self.assertLessEqual(len(selected), 4)

    def test_shortlist_caps_large_mandatory_liquidation_domains(self) -> None:
        mortgages = [OFFSETS["mortgage"] + index for index in range(20)]
        sell_property = OFFSETS["sell_prop"]
        legal = mortgages + [sell_property]
        scores = {action: float(index) for index, action in enumerate(legal)}
        selected = shortlist_actions(
            legal,
            mortgages[0],
            scores,
            legal,
            legal,
            limit=16,
        )
        self.assertEqual(len(selected), 16)
        self.assertEqual(selected[0], mortgages[0])
        self.assertIn(mortgages[-1], selected)
        self.assertIn(sell_property, selected)

    def test_asu_teacher_returns_legal_scored_candidates(self) -> None:
        self.env.phase = "post_roll"
        self.env.has_rolled = True
        self.env.players[0].position = 1
        action, candidates = asu_teacher_decision(self.env, 0)
        self.assertIn(action, self.env.get_allowed_actions(0))
        self.assertIn(action, candidates)
        self.assertLessEqual(len(candidates), 16)
        self.assertTrue(all("score" in value for value in candidates.values()))

    def test_asu_teacher_hash_covers_collection_config(self) -> None:
        self.assertEqual(
            asu_teacher_hash({"exploration_every": 9}),
            asu_teacher_hash({"exploration_every": 9}),
        )
        self.assertNotEqual(
            asu_teacher_hash({"exploration_every": 9}),
            asu_teacher_hash({"exploration_every": 10}),
        )

    def test_seeded_exploration_changes_behavior_but_not_teacher_label(self) -> None:
        candidates = {
            1: {"score": 3.0, "eligible": True, "forced": False, "mandatory": False},
            2: {"score": 2.0, "eligible": True, "forced": False, "mandatory": False},
            3: {"score": 1.0, "eligible": True, "forced": False, "mandatory": False},
        }
        behavior, exploratory = exploratory_behavior_action(
            1, candidates, seed=5, step=7, every=1, top_k=2
        )
        self.assertTrue(exploratory)
        self.assertIn(behavior, {2, 3})
        self.assertEqual(candidates[1]["score"], 3.0)

    def test_collection_uses_asu_labels_against_randomized_abc(self) -> None:
        seed = 21
        rows, report = collect_teacher_game(
            env=MonopolyEnv(agent_ids=[1], max_rounds=1),
            teacher_pid=1,
            opponents=scripted_opponents(seed, teacher_pid=1),
            game_id="smoke-21",
            seed=seed,
            teacher_bundle_hash=asu_teacher_hash({"exploration_every": 1}),
            exploration_every=1,
        )
        self.assertTrue(report["finished"])
        self.assertTrue(rows)
        self.assertEqual(report["teacher_policy"], "asu_value_v1")
        self.assertGreater(report["exploratory_actions"], 0)
        self.assertTrue(all(row["teacher_policy"] == "asu_value_v1" for row in rows))
        self.assertTrue(
            all(row["relabeled_action"] == row["teacher_action"] for row in rows)
        )
        self.assertTrue(all("asu_value_v1" not in row["prompt"] for row in rows))
        for row in rows:
            validate_dataset_row(row)

    def test_randomized_opponents_are_exactly_a_b_and_c(self) -> None:
        classes = {
            type(agent)
            for seed in range(20)
            for agent in scripted_opponents(seed, teacher_pid=0)
        }
        self.assertEqual({item.__name__ for item in classes}, {"TheHoarder", "TheDealMaker", "TheGambler"})

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
            teacher_policy="asu_value_v1",
            teacher_candidates={action: {"score": 1.0}},
            teacher_action=action,
            behavior_action=action,
            exploratory=False,
            relabeled_action=action,
            outcome="loss",
            teacher_bundle_hash="a" * 64,
        )
        validate_dataset_row(row)
        self.assertEqual(SCHEMA_VERSION, "monopoly-decision-v2")
        self.assertEqual(row["schema"], SCHEMA_VERSION)
        self.assertEqual(json.loads(row["completion"])["action"], "mortgage")
        self.assertEqual(row["state_hash"], sha256_text(row["prompt"]))

        broken = copy.deepcopy(row)
        broken["completion"] = '{"action":"end_turn"}'
        with self.assertRaisesRegex(ValueError, "relabeled action"):
            validate_dataset_row(broken)

        broken = copy.deepcopy(row)
        broken["exploratory"] = True
        with self.assertRaisesRegex(ValueError, "Exploration flag"):
            validate_dataset_row(broken)

        broken = copy.deepcopy(row)
        broken["candidate_scores"][str(action)] = 2.0
        with self.assertRaisesRegex(ValueError, "mismatched score"):
            validate_dataset_row(broken)

        broken = copy.deepcopy(row)
        other_action = next(item for item in row["legal_actions"] if item != action)
        broken["teacher_candidates"][str(other_action)] = {"score": 0.0}
        broken["candidate_scores"][str(other_action)] = 0.0
        broken["relabeled_action"] = other_action
        broken["completion"] = action_to_json(other_action, self.env, 0)
        with self.assertRaisesRegex(ValueError, "ASU labels"):
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
            teacher_policy="asu_value_v1",
            teacher_candidates={action: {"score": 1.0}},
            teacher_action=action,
            behavior_action=action,
            exploratory=False,
            relabeled_action=action,
            outcome="win",
            teacher_bundle_hash="b" * 64,
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

    def test_tokenizer_masks_gemma_generation_prompt_divergence(self) -> None:
        row = {"prompt": "state", "completion": '{"action":"end_turn"}'}
        tokenized = tokenize_rows([row], GemmaStyleTokenizer(), max_length=512)[0]
        response_start = next(
            index for index, label in enumerate(tokenized["labels"]) if label != -100
        )
        self.assertEqual(
            bytes(tokenized["input_ids"][:response_start]), b"<user>state<assistant>"
        )
        self.assertEqual(
            tokenized["labels"][response_start:],
            tokenized["input_ids"][response_start:],
        )

    def test_tokenizer_rejects_divergence_inside_user_context(self) -> None:
        row = {"prompt": "state", "completion": '{"action":"end_turn"}'}
        with self.assertRaisesRegex(ValueError, "no usable response boundary"):
            tokenize_rows([row], EarlyDivergenceTokenizer(), max_length=512)


if __name__ == "__main__":
    unittest.main()
