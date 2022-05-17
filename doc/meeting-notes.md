# Meeting notes
## 5-12
- Eli reupload data w/ nulls, available to help with code
- Hilly rerun ensemble with Hurtlex integrated
- Pangbo research more feasible way to combine features
- Avani run other ensemble networks for controversy task
- 

## 5-6
- Eli was able to extract NER from data & outputted to .csv file
    - how to store/format data
    - output vector representing counts for each type of named entity
- Hilly prepared tf-idf, punc counts, empathy ratings (could be useful for controversy)
    - 78% accuracy on dev data w/ SVM & features
- Pangbo extracted Hurtlex & tried merging w/ POS tagging
    - issues with Stanford POS tagger, so defaulted to NLTK POS tagger
    - encoding of Hurtlex in 17 dimensions
- Avani 
    - train classifier that takes inputs of ROBERTA & lexical features model
    - one model for all lexical features (concatenate representations into one vector space)
    - use logits as input to classifier (training how to weight outputs of roberta & other model)
    - normalize concatenated features
    - look to see if batch manager can also take care of features
- Report
    - spin up a brainstorming document on what we worked on
    - split who creates presentation & who writes on report
    - [Hilly] Discussion & Error Analysis
    - [Eli] Approach (your lexical feature)
    - [Pangbo] Approach (your lexical feature), Results
    - [Avani] System Architecture
- Presentation
    - [Pangbo] Approach w/ focus on lexical features
    - [Hilly] System Architecture, Issues & Success w/ focus on Error Analysis
    - [Eli] Approach w/ focus on approach for D2 section, Issues & Successes, Task Description
    - [Avani] slides maneuverer & support during QA

## 5-4
- able to run on GPU
- looked at POS
- fine-tuned model is ready (can turn in for D3)
- work on incorporating lexical features with random forest
    -  Pangbo will take a look at Hurtlex and how to preprocess
    -  Hilly can take count related lexical features
    -  Eli will look at NER
- Avani write ensembling script
- write short summary for each section of D3 report sumbission
- Hilly to fix extra quotations when reading csv input
- HIlly to do start exctended error analysis


## 5-1

- Fine-tuning RoBERTa model -- Eli found a tutorial.
- Look into how to get the GPU to work on Condor.
- (Need code to create model but can also just provide model.)
- Late fusion of classifiers in order to get lexical features?
    - Looks like it could be easier.
    - Might be better for the data.
- Features to add to second model:
    - TF-IDF
    - POS tagging
    - Punctuation counts
- Steps:
    1. Look into late fusion
    2. Talk to Haotian
    3. Get a skeleton up and going
- Other things to do:
    - Figure out how to get GPUs to work on Patas/Dryas
    - Look into what examples are/aren't being picked up

### To do:

- [Hilly] Save model and modify to print out the wrong sentences
- [Avani] Look into late fusion
- [Eli] Get GPU stuff working
- [Pangbo] Look into lexical feature classifiers

## 4-25
- Eli worked on some sections already
- Project Report Tasks @ 9:00am
    1) [Pangbo] to include table in results section & fill out the rest
        - no baseline to compare with
        - only trained on dev set
    2) [Eli] Approach
        - potentially fine-tuning LM
        - potentially using semantic features
        - potentially switching classifier layer to an RNN
        - using pre-trained LM b/c dataset is smol
    4) [Avani] Discussion
        - why are we considering the above three options in hopes of improving results
        - why we did what we did
        - error analysis (need to pretrain classifier layer! also, accuracy mimics positive example distribution in dataset. randomizing initial weights)
    6) [Hilly] Conclusion (few lines okay), Introduction
    7) [Eli] Abstract - just needs to be filled out
    8) Final review across all sections for typos, proper citations, any missing info as per D2 spec
      
- Model Tasks
    - [Avani] rerun d2 script on train data
    - [Hilly] debug results to make sure they are what we expect
    - [Eli] fine tune underlying LM on joke corpus (on one epoch)
    - [Avani] see if classifier layer can directly ingest embeddings (right after each batch is fed through as opposed to only after all batches have gone through) might not be necessary?
    - [Pangbo] read up on other approaches (semantic features, etc)

## 4-23
- test everything on condor
-   1) make sure conda environment can be recreated on dryas/patas
    - libcxx might cause problems, try running code on condor w/o it. if it works, we can just remove the requirement from the .yml file
    - remove your ling572 anaconda folder completely (rm -Rf [path_to_anaconda folder]
    - https://www.shane.st/teaching/575/spr22/patas-gpu.pdf <- follow to reinstall conda
    - google 'conda cheat sheet' for a list of helpful commands
-  2) set up condor file
    - request gpu!! 
- just use dev data for d2 and following parameters for baseline.py (parameters)
- python baseline.py --train_sentences data/hahackathon_prepo1_dev.csv --dev_sentences data/hahackathon_prepo1_dev.csv --hidden_layer 100 --learning_rate 0.001 --batch_size 64 --epochs 1 --output_file outputs/d2_out.txt
- write instructions on how to run everything
- in bash script, save metrics output to D2_score.out for deliverable checkin


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
