from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - lightweight local env
    torch = None
    nn = None

if torch is not None:
    from utils.schema_token_training import (
        TrainableTokenRowEmbedding,
        TrainableTokenRowLmHead,
        apply_schema_token_rows,
        are_input_output_embeddings_tied,
        enable_schema_token_row_training,
        initialize_token_rows_from_mean,
        save_schema_token_rows,
        schema_subtoken_ids,
        schema_token_ids,
        initialize_token_rows_from_subtokens,
    )


BaseModule = nn.Module if nn is not None else object


class FakeTokenizer:
    unk_token_id = 0

    def __init__(self) -> None:
        self.vocab = {"<unk>": 0, "hello": 1, "<headline>": 5, "</headline>": 6}

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def encode(self, token: str, add_special_tokens: bool = False) -> list[int]:
        mapping = {"<headline>": [2, 3, 4], "</headline>": [2, 7, 3, 4]}
        return mapping.get(token, [self.unk_token_id])


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class TinyModel(BaseModule):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        if tied:
            self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, module):
        self.embed = module

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, module):
        self.head = module


@unittest.skipIf(torch is None, "torch is not installed in this environment")
class SchemaTokenTrainingTest(unittest.TestCase):
    def test_schema_token_ids_reject_unknown_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(schema_token_ids(tokenizer, ["<headline>", "</headline>"]), [5, 6])
        with self.assertRaisesRegex(ValueError, "unk_token_id"):
            schema_token_ids(tokenizer, ["<missing>"])

    def test_schema_subtoken_ids_uses_old_tokenizer_decomposition(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(schema_subtoken_ids(tokenizer, ["<headline>"]), {"<headline>": [2, 3, 4]})

    def test_embedding_wrapper_only_has_trainable_delta_rows(self) -> None:
        base = nn.Embedding(8, 4)
        wrapped = TrainableTokenRowEmbedding(base, [5, 6])
        self.assertFalse(wrapped.base_embedding.weight.requires_grad)
        self.assertTrue(wrapped.delta.weight.requires_grad)

        output = wrapped(torch.tensor([[1, 5, 6]])).sum()
        output.backward()
        self.assertIsNone(wrapped.base_embedding.weight.grad)
        self.assertIsNotNone(wrapped.delta.weight.grad)

    def test_lm_head_wrapper_only_changes_selected_token_logits(self) -> None:
        base = nn.Linear(4, 8, bias=False)
        wrapped = TrainableTokenRowLmHead(base, [5, 6])
        with torch.no_grad():
            wrapped.delta_weight[0].fill_(1.0)
        hidden = torch.ones(2, 3, 4)
        base_logits = base(hidden)
        wrapped_logits = wrapped(hidden)
        self.assertTrue(torch.allclose(wrapped_logits[..., :5], base_logits[..., :5]))
        self.assertTrue(torch.allclose(wrapped_logits[..., 7:], base_logits[..., 7:]))
        self.assertTrue(torch.allclose(wrapped_logits[..., 5], base_logits[..., 5] + 4.0))

    def test_enable_schema_token_row_training_wraps_untied_lm_head(self) -> None:
        model = TinyModel(tied=False)
        self.assertFalse(are_input_output_embeddings_tied(model))
        enable_schema_token_row_training(model, [5, 6])
        self.assertIsInstance(model.get_input_embeddings(), TrainableTokenRowEmbedding)
        self.assertIsInstance(model.get_output_embeddings(), TrainableTokenRowLmHead)

    def test_enable_schema_token_row_training_skips_tied_lm_head_wrapper(self) -> None:
        model = TinyModel(tied=True)
        self.assertTrue(are_input_output_embeddings_tied(model))
        enable_schema_token_row_training(model, [5, 6])
        self.assertIsInstance(model.get_input_embeddings(), TrainableTokenRowEmbedding)
        self.assertNotIsInstance(model.get_output_embeddings(), TrainableTokenRowLmHead)

    def test_initialize_token_rows_from_mean_sets_input_and_output_rows(self) -> None:
        model = TinyModel(tied=False)
        with torch.no_grad():
            model.embed.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4))
            model.head.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4) + 100)
        initialize_token_rows_from_mean(model, [5, 6])
        expected_input = torch.arange(20, dtype=torch.float32).view(5, 4).mean(dim=0)
        expected_output = (torch.arange(20, dtype=torch.float32).view(5, 4) + 100).mean(dim=0)
        self.assertTrue(torch.allclose(model.embed.weight[5], expected_input))
        self.assertTrue(torch.allclose(model.embed.weight[6], expected_input))
        self.assertTrue(torch.allclose(model.head.weight[5], expected_output))
        self.assertTrue(torch.allclose(model.head.weight[6], expected_output))

    def test_initialize_token_rows_from_subtokens_sets_input_and_output_rows(self) -> None:
        model = TinyModel(tied=False)
        with torch.no_grad():
            model.embed.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4))
            model.head.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4) + 100)
        initialize_token_rows_from_subtokens(
            model,
            token_ids=[5, 6],
            token_to_subtoken_ids={"<headline>": [2, 3, 4], "</headline>": [2, 7, 3, 4]},
            tokens=["<headline>", "</headline>"],
        )
        expected_input = torch.arange(32, dtype=torch.float32).view(8, 4)[[2, 3, 4]].mean(dim=0)
        expected_output = (torch.arange(32, dtype=torch.float32).view(8, 4) + 100)[[2, 3, 4]].mean(dim=0)
        self.assertTrue(torch.allclose(model.embed.weight[5], expected_input))
        self.assertTrue(torch.allclose(model.head.weight[5], expected_output))

    def test_save_and_apply_schema_token_rows(self) -> None:
        model = TinyModel(tied=False)
        enable_schema_token_row_training(model, [5, 6])
        with torch.no_grad():
            model.embed.delta.weight.fill_(2.0)
            model.head.delta_weight.fill_(3.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema_token_rows.pt"
            save_schema_token_rows(model, ["<headline>", "</headline>"], path)

            target = TinyModel(tied=False)
            old_input = target.embed.weight.detach().clone()
            old_output = target.head.weight.detach().clone()
            apply_schema_token_rows(target, path)
            self.assertFalse(torch.allclose(target.embed.weight[5], old_input[5]))
            self.assertFalse(torch.allclose(target.head.weight[5], old_output[5]))
            self.assertTrue(torch.allclose(target.embed.weight[1], old_input[1]))
            self.assertTrue(torch.allclose(target.head.weight[1], old_output[1]))


if __name__ == "__main__":
    unittest.main()
