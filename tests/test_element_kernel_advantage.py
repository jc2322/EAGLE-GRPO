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


def completion(headline: str) -> str:
    return (
        "<thinking>choose clean product layout</thinking>"
        "<final_prompt>"
        "<preservation_constraints>keep exact product</preservation_constraints>"
        "<product_presentation>large centered product</product_presentation>"
        f"<headline>{headline}</headline>"
        "<subtitle></subtitle>"
        "<feature_callouts></feature_callouts>"
        "<specification_information></specification_information>"
        "<quantity_bundle_information></quantity_bundle_information>"
        "<variant_information></variant_information>"
        "<price_promotion_information></price_promotion_information>"
        "<seller_logistics_information></seller_logistics_information>"
        "<trust_information></trust_information>"
        "<usage_information></usage_information>"
        "<comparison_visual></comparison_visual>"
        "<supporting_visuals></supporting_visuals>"
        "<composition_layout>balanced square layout</composition_layout>"
        "<scene_style>bright studio background</scene_style>"
        "<lighting_rendering>soft commercial shadows</lighting_rendering>"
        "<negative_constraints>no invented product</negative_constraints>"
        "</final_prompt>"
    )


def padded_char_batch(texts):
    import torch

    max_len = max(len(text) for text in texts)
    ids = torch.zeros((len(texts), max_len), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for row, text in enumerate(texts):
        values = [ord(char) for char in text]
        ids[row, : len(values)] = torch.tensor(values, dtype=torch.long)
        mask[row, : len(values)] = 1
    return ids, mask


@unittest.skipUnless(TORCH_AVAILABLE, "torch is installed in the training environment, not this local lint environment")
class ElementKernelAdvantageTest(unittest.TestCase):
    def test_single_element_mapping_conserves_standard_image_advantage(self) -> None:
        import torch

        from rewards.online_component_reward import ComponentReward, ComponentRewardBatch, ComponentRewardStore
        from rewards.online_component_reward import normalize_component_advantages
        from utils.advantage_protocol import AdvantageContext
        from utils.element_kernel_advantage import ElementKernelAdvantageProvider

        texts = [completion("Clean Mug"), completion("Premium Mug"), completion("Gift Ready Mug")]
        rewards = [
            ComponentReward(
                request_id=f"sample-{index}",
                row_key="item-a",
                sample_index=index,
                raw_completion=text,
                schema_valid=True,
                active=True,
                cot_reward=0.5,
                image_reward=float(index),
                scalar_reward=float(index),
            )
            for index, text in enumerate(texts)
        ]
        normalize_component_advantages(rewards, min_successful_samples=1, min_successful_ratio=0.0)
        store = ComponentRewardStore()
        store.set_latest(ComponentRewardBatch(rewards))
        ids, mask = padded_char_batch(texts)
        provider = ElementKernelAdvantageProvider(
            store,
            config={
                "element_credit": {
                    "element_tags": ["headline"],
                    "eta": 1.0,
                    "lambda": 0.01,
                    "include_thinking": False,
                    "condition_number_max": 1_000_000.0,
                }
            },
        )

        advantages = provider.token_advantages(
            AdvantageContext(
                completion_ids=ids,
                completion_mask=mask,
                sequence_advantages=torch.zeros(len(texts)),
                tokenizer=CharTokenizer(),
            )
        )

        self.assertEqual(len(provider.last_diagnostics), 1)
        self.assertEqual(provider.last_diagnostics[0]["fallback"], "")
        for row, reward in enumerate(rewards):
            selected = provider.last_loss_mask[row].bool()
            self.assertTrue(selected.any())
            self.assertAlmostEqual(float(advantages[row][selected].mean().item()), reward.image_advantage, places=5)
            constraint_start = texts[row].index("keep exact product")
            constraint_end = constraint_start + len("keep exact product")
            self.assertTrue(torch.all(advantages[row, constraint_start:constraint_end].eq(0.0)))

    def test_keeps_cot_advantage_on_thinking_tokens(self) -> None:
        import torch

        from rewards.online_component_reward import ComponentReward, ComponentRewardBatch, ComponentRewardStore
        from rewards.online_component_reward import normalize_component_advantages
        from utils.advantage_protocol import AdvantageContext
        from utils.element_kernel_advantage import ElementKernelAdvantageProvider

        text = completion("Clean Mug")
        rewards = [
            ComponentReward(
                request_id="sample-0",
                row_key="item-a",
                sample_index=0,
                raw_completion=text,
                schema_valid=True,
                active=True,
                cot_reward=1.0,
                image_reward=1.0,
                scalar_reward=1.0,
                cot_advantage=2.0,
                image_advantage=3.0,
            )
        ]
        normalize_component_advantages(rewards, min_successful_samples=1, min_successful_ratio=0.0)
        rewards[0].cot_advantage = 2.0
        rewards[0].image_advantage = 3.0
        store = ComponentRewardStore()
        store.set_latest(ComponentRewardBatch(rewards))
        ids, mask = padded_char_batch([text])
        provider = ElementKernelAdvantageProvider(
            store,
            config={"element_credit": {"element_tags": ["headline"], "include_thinking": True}},
        )

        advantages = provider.token_advantages(
            AdvantageContext(
                completion_ids=ids,
                completion_mask=mask,
                sequence_advantages=torch.zeros(1),
                tokenizer=CharTokenizer(),
            )
        )

        start = text.index("choose clean")
        end = start + len("choose clean product layout")
        self.assertTrue(torch.all(advantages[0, start:end].eq(2.0)))


if __name__ == "__main__":
    unittest.main()
