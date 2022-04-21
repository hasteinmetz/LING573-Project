# Meeting notes
## 4-20

- Dev data is not a sample of test and train
  - Eli will replit to 80-10-10 and push to main
- cite tutorials in code
- RNN Classifier
  - how to get the RNN to work with pytorch native implementation
  - gradient descent updating every parameter ?
  - Pangbo can ask classmate who is knowledgeable about pytorch
  - most recent stuff is in hilly (branch name)
- Embeddings
  - works on a single input sentencing, updating now to work on a list of input strings
  - will output embeddings to a separate file (size will be [768])
  - will push polished version by EOD 4/20
- Swap RNN for simple Neural Network
  - check out article from Pangbo
- Conda on Dryas
  - copy-paste environment file & use instructions from Shane
- Next sync 4/23 Saturday 5:30pm 
  - finish our subtasks by then
  - finish remaining documentation (overleaf, readme)
  - put everything together and make sure it runs

## 4-19

- Moving to Hyak (some overhead so might be tough)
  - Git not working :/ but stick with it for not
- How to divide up the dev set -- reach out to Gina and Haotian about th    is.
  - For now switching to 80/10/10
- Getting rid of staging to be just main branch

## 4-14

- Have embeddings... need to decide evaluation.
- Split up coding:
  - Create train, dev, & test sets
  - What level of preprocessing produces the right results?
    - Are there going to be a lot of unknown tokens?
  - Make embeddings from RoBERTA
  - Train a classifier
    - Get SVM set up and then try to get RNN
    - Maybe just add a on-layer classifier layer on top? (This we'd need to train)
  - Evaluate the classifier
- Other work:
  - Set up Anaconda environment
  - Ask Brandon for shared space
  - Decide on collaboration software
- Dividing up the work:
  - Eli: split up data and preprocessing
  - Avani: forward embeddings from the text
  - Hilly: one-layer classifier
  - Pangbo: evaluation & SVM

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
