from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


class CharTokenizer:
    def decode(self, ids, skip_special_tokens: bool = False):
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in ids)

    def __call__(self, text, **kwargs):
        del kwargs
        return {"offset_mapping": [(index, index + 1) for index, _char in enumerate(text)]}


@unittest.skipUnless(TORCH_AVAILABLE, "torch is installed in the training environment, not this local lint environment")
class DecoupledAdvantageTest(unittest.TestCase):
    def test_masks_thinking_and_non_constraint_final_prompt_content_only(self) -> None:
        import torch

        from rewards.online_component_reward import ComponentReward, ComponentRewardBatch, ComponentRewardStore
        from utils.advantage_protocol import AdvantageContext
        from utils.decoupled_advantage import DecoupledAdvantageProvider

        text = """<thinking>choose clean design</thinking><final_prompt><preservation_constraints>keep exact product</preservation_constraints><product_presentation>large hero</product_presentation><headline>Great Mug</headline><subtitle></subtitle><feature_callouts></feature_callouts><specification_information></specification_information><quantity_bundle_information></quantity_bundle_information><variant_information></variant_information><price_promotion_information></price_promotion_information><seller_logistics_information></seller_logistics_information><trust_information></trust_information><usage_information></usage_information><comparison_visual></comparison_visual><supporting_visuals></supporting_visuals><composition_layout>center layout</composition_layout><scene_style>bright studio</scene_style><lighting_rendering>soft shadows</lighting_rendering><negative_constraints>no extra items</negative_constraints></final_prompt>"""
        ids = torch.tensor([[ord(char) for char in text]])
        mask = torch.ones_like(ids)
        store = ComponentRewardStore()
        store.set_latest(
            ComponentRewardBatch(
                [
                    ComponentReward(
                        request_id="r0",
                        row_key="item-a",
                        sample_index=0,
                        raw_completion=text,
                        schema_valid=True,
                        cot_advantage=1.5,
                        image_advantage=2.5,
                        active=True,
                    )
                ]
            )
        )
        provider = DecoupledAdvantageProvider(store)
        advantages = provider.token_advantages(
            AdvantageContext(
                completion_ids=ids,
                completion_mask=mask,
                sequence_advantages=torch.tensor([0.0]),
                tokenizer=CharTokenizer(),
            )
        )
        decoded = text
        self.assertTrue(torch.all(advantages[0, decoded.index("choose") : decoded.index("design") + len("design")].eq(1.5)))
        self.assertTrue(torch.all(advantages[0, decoded.index("large hero") : decoded.index("large hero") + len("large hero")].eq(2.5)))
        self.assertTrue(torch.all(advantages[0, decoded.index("keep exact product") : decoded.index("keep exact product") + len("keep exact product")].eq(0.0)))
        self.assertTrue(torch.all(advantages[0, decoded.index("no extra items") : decoded.index("no extra items") + len("no extra items")].eq(0.0)))
        self.assertEqual(float(provider.last_loss_mask.sum().item()), float(advantages.ne(0).sum().item()))


if __name__ == "__main__":
    unittest.main()
