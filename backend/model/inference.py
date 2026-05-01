"""
mBART inference wrapper for formality-conditioned EN→KR translation.

Loads a fine-tuned checkpoint and runs beam search to produce Korean output.
The checkpoint already has formality tokens in the tokenizer and the correct
generation_config (forced_bos_token_id=ko_KR, no_repeat_ngram_size, etc.)
saved from training — no re-initialization needed.

Usage:
    inf = MBartInference("models/noonchi-mbart")
    korean = inf.translate("Can you help me?", "formal")
"""

import logging

import torch
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

logger = logging.getLogger(__name__)


class MBartInference:
    def __init__(self, model_dir: str):
        logger.info(f"Loading mBART checkpoint from {model_dir}")
        self.tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
        self.model = MBartForConditionalGeneration.from_pretrained(model_dir)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        logger.info(f"mBART loaded on {self.device}")

    def translate(self, text: str, formality: str) -> str:
        """
        Args:
            text: English source sentence
            formality: 'formal' | 'polite' | 'casual'
        Returns:
            Korean translation at the specified speech level
        """
        src = f"<{formality}> {text}"
        inputs = self.tokenizer(
            src, return_tensors="pt", max_length=128, truncation=True
        ).to(self.device)

        # forced_bos_token_id is stored in generation_config from training —
        # do NOT pass it again here; transformers >=4.46 raises ValueError on conflict.
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=4,
                max_new_tokens=128,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
