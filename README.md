After installing requirements you need to install the english language for spaCy

```bash
python -m spacy download en_core_web_sm
```

# TODO

- Use regex on sentences to see if a double linebreak should be present. Ie if first line doesn't end with a full stop or the next line starts with a lowercase letter then we should probably remove the linebreak so that Spacy doesn't distinguish it as two seperate sentences
