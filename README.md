After installing requirements you need to install the english language for spaCy

```bash
python -m spacy download en_core_web_sm
```

# TODO
1. Create a larger training dataset for NER of ingredients
2. Create a script (w regex) to preprocess ingredient text
    - Any amount + units joined together need to be split ie 450ml -> 450 ml
    - All lowercase
3. Create a model to identify topic of line of text
4. Use regex on sentences to see if a double linebreak should be present. Ie if first line doesn't end with a full stop or the next line starts with a lowercase letter then we should probably remove the linebreak so that Spacy doesn't distinguish it as two seperate sentences
