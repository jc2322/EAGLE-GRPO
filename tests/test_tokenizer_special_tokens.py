from __future__ import annotations

import unittest

from prompts.tokenizer_special_tokens import add_schema_special_tokens, schema_special_tokens
from utils.modeling import decode_preserving_schema_tags


class FakeTokenizer:
    def __init__(self) -> None:
        self.additional_special_tokens: list[str] = []

    def add_special_tokens(self, payload: dict[str, list[str]]) -> int:
        tokens = payload["additional_special_tokens"]
        self.additional_special_tokens.extend(tokens)
        return len(tokens)


class DecodeTokenizer:
    all_special_tokens = ["<pad>", "<headline>", "</headline>"]

    def decode(self, ids, skip_special_tokens: bool = False):
        del ids
        text = "<pad><headline>Title</headline>"
        if skip_special_tokens:
            return "Title"
        return text


class TokenizerSpecialTokensTest(unittest.TestCase):
    def test_schema_special_tokens_include_open_and_close_tags(self) -> None:
        tokens = schema_special_tokens()
        self.assertIn("<thinking>", tokens)
        self.assertIn("</thinking>", tokens)
        self.assertIn("<final_prompt>", tokens)
        self.assertIn("</final_prompt>", tokens)
        self.assertIn("<seller_logistics_information>", tokens)
        self.assertIn("</seller_logistics_information>", tokens)
        self.assertEqual(len(tokens), len(set(tokens)))

    def test_add_schema_special_tokens_is_idempotent_for_existing_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        first = add_schema_special_tokens(tokenizer)
        second = add_schema_special_tokens(tokenizer)
        self.assertGreater(first, 0)
        self.assertEqual(second, 0)


    def test_decode_preserving_schema_tags_keeps_schema_boundaries(self) -> None:
        decoded = decode_preserving_schema_tags(DecodeTokenizer(), [1, 2], skip_special_tokens=True)
        self.assertEqual(decoded, "<headline>Title</headline>")


if __name__ == "__main__":
    unittest.main()
