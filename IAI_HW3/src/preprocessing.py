"""Versioned preprocessing shared by training, evaluation and prediction."""

import re

from .eda import preprocess

MODES = ("whitespace-v1", "url-http-v1", "emoji-short-v1")
# Fixed descriptive phrases, not class labels; unsupported sequences stay unchanged.
EMOJI_WORDS = {
    '🙏': 'folded hands', '✈': 'airplane', '😡': 'angry face',
    '😭': 'crying face', '👍': 'thumbs up', '😢': 'crying face',
    '😊': 'smiling face', '👏': 'clapping hands', '❤': 'heart',
    '😂': 'tears of joy', '✌': 'victory hand', '🆘': 'sos',
    '👎': 'thumbs down', '👌': 'ok hand', '😄': 'smiling face',
    '💙': 'blue heart', '😉': 'winking face', '😃': 'smiling face',
    '💔': 'broken heart', '😤': 'face with steam from nose',
    '☺': 'smiling face', '😍': 'heart eyes', '😞': 'disappointed face',
    '😔': 'pensive face', '😠': 'angry face', '😒': 'unamused face',
    '😩': 'weary face', '😫': 'tired face', '😕': 'confused face',
    '😐': 'neutral face', '😎': 'smiling face with sunglasses',
    '😘': 'blowing a kiss', '✅': 'check mark', '❌': 'cross mark',
}
_EMOJI_BASE = r'[\U0001F000-\U0001FAFF\u2600-\u27BF]'
_EMOJI_UNIT = _EMOJI_BASE + r'[\ufe0e\ufe0f]?[\U0001F3FB-\U0001F3FF]?'
EMOJI_SEQUENCE = re.compile(_EMOJI_UNIT + r'(?:\u200d' + _EMOJI_UNIT + r')*')


def describe_emoji(text):
    previous_end, previous_symbol = -1, None

    def replace(match):
        nonlocal previous_end, previous_symbol
        raw = match.group()
        symbol = raw.rstrip('\ufe0e\ufe0f')
        phrase = EMOJI_WORDS.get(symbol)
        repeated = match.start() == previous_end and symbol == previous_symbol
        previous_end, previous_symbol = match.end(), symbol
        if phrase is None:
            return raw
        return ' ' if repeated else ' ' + phrase + ' '

    return EMOJI_SEQUENCE.sub(replace, text)
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
    elif mode == "emoji-short-v1":
        text = describe_emoji(text)
    return preprocess(text)
