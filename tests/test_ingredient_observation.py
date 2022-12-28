import pytest
from utils.text_processing import process_text, is_ingredient_sent
import spacy


def test_ingredient_match():
    text = "45ml/3 tablespoons olive oil"
    doc = process_text(text)
    assert is_ingredient_sent(doc, is_paragraph=False)
