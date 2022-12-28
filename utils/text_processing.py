import spacy
from spacy.language import Language
import re


def process_text(text: str) -> spacy.language.Doc:
    nlp = spacy.load("en_core_web_sm")

    @Language.component("set_custom_boundaries")
    def set_custom_boundaries(doc):
        for token in doc[:-1]:
            if token.text == "\n\n":
                doc[token.i].is_sent_start = True
        return doc

    nlp.add_pipe("set_custom_boundaries", before="parser")
    return nlp(text)


def is_ingredient_sent(paragraph: list, is_paragraph: bool = True) -> bool:
    if is_paragraph:
        # Ingredients only have one Span in the paragraph
        if len(paragraph) > 1:
            return False

        ingredient_span = paragraph[0]
    else:
        ingredient_span = paragraph

    print(ingredient_span.text)

    print(
        [(token.text, token.pos_, token.dep_, token.tag_) for token in ingredient_span]
    )
    # Only Ingredient's begin with a number
    if list(ingredient_span)[0].pos_ == "NUM":
        return True

    # Ingredients often follow the format of: [AMOUNT] of [optional adjetive] [INGREDIENT] eg pinch of salt, ROOT prep ... pobj
    serving_sentence_list = list(ingredient_span)
    if (
        serving_sentence_list[0].dep_ == "ROOT"
        and serving_sentence_list[1].dep_ == "prep"
        and serving_sentence_list[-1].dep_ == "pobj"
    ):
        return True

    # Ingredient sometimes listed with some extra steps but no additional nouns. Eg "salt, to taste"
    check_if_noun = lambda token: token.pos_ == "NOUN"
    if serving_sentence_list[0].pos_ == "NOUN" and not any(
        [check_if_noun(token) for token in serving_sentence_list[1:]]
    ):
        return True

    return False
