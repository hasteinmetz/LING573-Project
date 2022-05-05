# Lab Notebook

Use this file to document any important changes you've made or any records of running the system.

This notebook uses Markdown. Use '#' to create headings. When you make a change put the heading ### followed by the date and your name. Always add changes above previous changes. Leave these instructions on top. You can also use - to make a list of items and \[text\](link) to add a link.

### 5-4 Pangbo
- Checked HurtBERT and Lexical Feature on QA (https://downloads.hindawi.com/journals/complexity/2021/2893257.pdf) paper. The accuracy on Ubuntu dataset is Random Forest > SVC > Logistic Regression > LinearSVC > MultinomialNB
- Learned that features output to RF(https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) need to be: a) of impact; b) as unrelated as possible
- I. Lexical feature candicates (2021 semieval) for BoW:
    1. punctuation counts
    2. letter counts
    3. question mark counts
    4. wh-word counts(?) ([who, whose, what, when, which, why, where, how])
    5. hurtlex counts(?) (needs to be filtered on what catogories to be kept)
    6. personal pronoun counts(?) (count # of PRP from the result of POS tagging)
    7. Named Entity Marking (https://www.nltk.org/api/nltk.chunk.html; taking POS tagging as input; tutorial: https://machinelearningknowledge.ai/beginners-guide-to-named-entity-recognition-ner-in-nltk-library-python/#:~:text=NLTK%20provides%20a%20function%20nltk,ever%20it%20founded%20named%20entity.)
  
  II. Lexical feature candidates for direct input:
    1. POS tagging (https://www.nltk.org/api/nltk.tag.html; using Penn Treebank)

### 4-11 Hilly
- Read the ACL paper on shared task--they used BERT as a baseline.
- Found out that [TensorFlow](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) and [PyTorch](https://pytorch.org/hub/huggingface_pytorch-transformers/) have pre-built BERT models.

### 4-7 Hilly 

- Created lab notebook and created a data folder in the src directory.
- Tagged [deliverable 1](https://github.com/hasteinmetz/LING573-Project/releases/tag/D1).
