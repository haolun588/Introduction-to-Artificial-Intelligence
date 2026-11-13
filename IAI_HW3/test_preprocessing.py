import unittest
from pathlib import Path

from preprocessing import prepare_text
from training import TextDataset
from transformers import AutoTokenizer


class PreprocessingTests(unittest.TestCase):
    def test_urls_and_surrounding_sentiment(self):
        self.assertEqual(prepare_text("Not good! (HTTPS://t.co/AbC). #fail @air www.test.com/x?x=1&y=2!", "url-http-v1"),
                         "Not good! (http). #fail @air http!")
        text = "I can't fly :( @air #delay \U0001f622"
        self.assertEqual(prepare_text(text, "url-http-v1"), text)

    def test_legacy_and_idempotence(self):
        text = "  @air\nhttp://t.co/a  bad! "
        self.assertEqual(prepare_text(text), "@air http://t.co/a bad!")
        transformed = prepare_text(text, "url-http-v1")
        self.assertEqual(prepare_text(transformed, "url-http-v1"), transformed)
        with self.assertRaises(ValueError):
            prepare_text(text, "unknown")

    def test_train_and_prediction_use_same_tokens(self):
        tokenizer = AutoTokenizer.from_pretrained(Path(__file__).parent / "outputs/base_model", local_files_only=True)
        self.assertEqual(tokenizer.tokenize("http"), ["http"])
        rows = [{"id": "1", "text": "bad https://t.co/a", "label": "negative"}]
        labeled = TextDataset(rows, tokenizer, 64, preprocessing="url-http-v1")[0]
        prediction = TextDataset(rows, tokenizer, 64, labeled=False, preprocessing="url-http-v1")[0]
        self.assertEqual(labeled.pop("labels"), 0)
        self.assertEqual(labeled, prediction)
        self.assertEqual(prediction["input_ids"], tokenizer("bad http")["input_ids"])


if __name__ == "__main__":
    unittest.main()
