# Meeting notes

## 4-14

### Pre-meeting

#### Hilly:

- Need to divide data (one program).
- Have embeddings... need to decide evaluation.
- Split up coding:
  - Create train, dev, & test sets
  - Make embeddings from RoBERTA
  - Train a classifier
  - Evaluate the classifier
- Other work:
  - Set up Anaconda environment
  - Ask Brandon for shared space
  - Decide on collaboration software

## 4-11

- Decided on a baseline task
  - RoBERTa
  - Use PyTorch since out of the box
  - Still room to add additional features and improve performance (e.g., punctuation frequencies)
  - Use provided tokenizer
  - Backup: Word2Vec embeddings and SVM
    - Date of pivot: Check if on the right track next Tuesday
- Splitting up work:
  - Split into 4 tasks to build RoBERTa model -- first do some research
  - Maybe split it up so that 2 people write the program, 1 person writes the report, and one person does research
- Workflow:
  - Work on a branch with the name "alias/purpose" for now
  - Regroup later on
  - Staging branch for development and main is for deliverable tagging

### Action items

- Go through a [tutorial](https://pytorch.org/hub/huggingface_pytorch-transformers/#using-modelforsequenceclassification-to-do-paraphrase-classification-with-bert)
  - Think about how it can be split
  - Think about interoperability
- Meet back on Thursday to talk about how to split up work
