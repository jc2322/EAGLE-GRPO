from __future__ import annotations

import unittest

from utils.completion_schema import FINAL_PROMPT_TAGS, parse_completion


STRUCTURED = """
<thinking>
one product, preserve logo, choose clean layout, avoid unsupported claims.
</thinking>
<final_prompt>
  <preservation_constraints>keep the exact product</preservation_constraints>
  <product_presentation>center the product</product_presentation>
  <headline>MODEL X</headline>
  <subtitle></subtitle>
  <feature_callouts>one grounded callout</feature_callouts>
  <specification_information></specification_information>
  <quantity_bundle_information></quantity_bundle_information>
  <variant_information></variant_information>
  <price_promotion_information></price_promotion_information>
  <seller_logistics_information></seller_logistics_information>
  <trust_information></trust_information>
  <usage_information></usage_information>
  <comparison_visual></comparison_visual>
  <supporting_visuals></supporting_visuals>
  <composition_layout>centered hero</composition_layout>
  <scene_style>clean studio</scene_style>
  <lighting_rendering>soft light</lighting_rendering>
  <negative_constraints>no extra product</negative_constraints>
</final_prompt>
""".strip()


class CompletionSchemaTest(unittest.TestCase):
    def test_structured_completion_exposes_all_element_spans(self) -> None:
        parsed = parse_completion(STRUCTURED, require_structured_children=True)
        self.assertEqual(set(parsed.final_prompt_spans), set(FINAL_PROMPT_TAGS))
        self.assertIn("centered hero", parsed.final_prompt_spans["composition_layout"].text)
        span = parsed.final_prompt_spans["composition_layout"]
        self.assertEqual(parsed.raw_text[span.start : span.end], span.text)

    def test_current_think_format_is_backward_compatible(self) -> None:
        parsed = parse_completion("<think>short reasoning</think><final_prompt>usable prompt</final_prompt>")
        self.assertEqual(parsed.reasoning, "short reasoning")
        self.assertEqual(parsed.final_prompt, "usable prompt")

    def test_new_thinking_format_is_supported(self) -> None:
        parsed = parse_completion("<thinking>short reasoning</thinking><final_prompt>usable prompt</final_prompt>")
        self.assertEqual(parsed.reasoning, "short reasoning")
        self.assertEqual(parsed.final_prompt, "usable prompt")

    def test_missing_final_prompt_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "final_prompt"):
            parse_completion("<reasoning>analysis</reasoning>")


if __name__ == "__main__":
    unittest.main()
