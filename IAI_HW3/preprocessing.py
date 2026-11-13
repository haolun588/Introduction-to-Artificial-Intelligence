"""Versioned preprocessing shared by training, evaluation and prediction."""

import re

from eda import preprocess

MODES = ("whitespace-v1", "url-http-v1")
URL = re.compile(r"\b(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
TRAILING_PUNCTUATION = ".,!?;:)]}\"'"


def prepare_text(text, mode="whitespace-v1"):
    if mode not in MODES:
        raise ValueError(f"Unsupported preprocessing: {mode}")
    if mode == "url-http-v1":
        def replace(match):
            raw = match.group()
            core = raw.rstrip(TRAILING_PUNCTUATION)
            return "http" + raw[len(core):]
        text = URL.sub(replace, text)
    return preprocess(text)
