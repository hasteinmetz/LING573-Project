# Lab Notebook

Use this file to document any important changes you've made or any records of running the system.

This notebook uses Markdown. Use '#' to create headings. When you make a change put the heading ### followed by the date and your name. Always add changes above previous changes. Leave these instructions on top. You can also use - to make a list of items and \[text\](link) to add a link.

## 5-20 Neural Kfold Ensemble performances (_test set_)

- **Config**: {'lr': 2e-05, 'batch_size': 32, 'hidden_size': 40, 'output_size': 25, 'dropout_mlp': 0.4, 'dropout_roberta': 0.16, 'epochs': 5}
- K-Best:
	- humor: 
		- f1: {'f1': 0.9585439838220424}
		- accuracy: {'accuracy': 0.9488139825218477}
	- controversy:
		- f1: {'f1': 0.5158730158730158}
		- accuracy: {'accuracy': 0.5060728744939271}
- PCA:
	- humor: 
		- f1: {'f1': 0.9557157569515962}
		- accuracy: {'accuracy': 0.9463171036204744}
	- controversy:
		- f1: {'f1': 0.4304932735426009}
		- accuracy: {'accuracy': 0.48582995951417}

## 5-27 Neural Ensemble performances (dev set)

### Vanilla neural: 
- **Config**: {'lr': 2e-05, 'batch_size': 32, 'hidden_size': 40, 'output_size': 25, 'dropout_mlp': 0.4, 'dropout_roberta': 0.16, 'epochs': 5}
- K-Best:
	- humor: 
		- f1: {'f1': 0.9366834170854271}
		- accuracy: {'accuracy': 0.92125}
	- controversy:
		- f1: {'f1': 0.45879732739420936}
		- accuracy: {'accuracy': 0.5070993914807302}
- PCA:
	- humor: 
		- f1: {'f1': 0.9294605809128631}
		- accuracy: {'accuracy': 0.915}
	- controversy:
		- f1: {'f1': 0.4567627494456763}
		- accuracy: {'accuracy': 0.5030425963488844}
### neural w/ kfolds:
	- **Config**: {'lr_transformer': 2e-05, 'lr_classifier': 0.008, 'lr_regressor': 0.01, 'batch_size': 32, 'kfolds': 3, 'epochs': 3, 'hidden_size': 40, 'dropout_mlp': 0.4, 'dropout_roberta': 0.165}
	- K-Best:
		- humor: 
			- f1: {'f1': 0.9442691903259726}
			- accuracy: {'accuracy': 0.93375}
		- controversy:
			- _f1: {'f1': 0.4915254237288136}_
			- _accuracy: {'accuracy': 0.513184584178499} _
	- PCA:
		- humor: 
			- _f1: {'f1': 0.9465968586387434}_
			- _accuracy: {'accuracy': 0.93625}_
		- controversy:
			- f1: {'f1': 0.47391304347826085}
			- accuracy: {'accuracy': 0.5091277890466531} 

## 5-26 Neural Ensemble performances (dev set)

- Vanilla neural:
	- K-Best:
		- humor: 
			- f1: {'f1': 0.9326530612244897} | accuracy: {'accuracy': 0.9175}
		- controversy:
			- f1: {'f1': 0.0} | accuracy: {'accuracy': 0.4969574036511156}
	- PCA:
		- humor: 
			- f1: {'f1': 0.9156398104265402} | accuracy: {'accuracy': 0.88875}
		- controversy:
			- f1: {'f1': 0.0} | accuracy: {'accuracy': 0.4969574036511156}
- neural w/ kfolds:
	- K-Best:
		- humor: 
			- f1: {'f1': 0.9404517453798769} | accuracy: {'accuracy': 0.9275}
		- controversy:
			- f1: {'f1': 0.5246548323471399} | accuracy: {'accuracy': 0.5111561866125761} 
	- PCA:
		- humor: 
			- f1: {'f1': 0.9454905847373637} | accuracy: {'accuracy': 0.93125}
		- controversy:
			- f1: {'f1': 0.5010266940451744} | accuracy: {'accuracy': 0.5070993914807302} 
	- **In both there are higher f1 scores in the earlier epochs... it seems to be unstable though... the transformer is not able to learn more than guessing the majority group**

### 5-9 Avani
- v1 ensemble 
	- 100 estimator rf
- v2 ensemble
	- using configuration from src/configs/random_forest_v2.json

### 5-11 Hilly

- Built a ensemble model using PyTorch and k-fold cross-validation learning. It does not have HurtLex incorporated in it yet. 
- Initial run found that the score was lower ~91% on the dev set.
    - Lowered the classifier learning rate to 5e-2 and changed dropout to 0.5 for first layer and 0.5 for second layer
    - Accuracy: 0.93875
    - F1: 0.9501

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

### 5-1 Hilly

- Got the following working:
  - Randomization
  - Model is now learning and generating output
- Current results:
  - SVM: ~89%
  - Neural network (scikit): ~90%
  - Neural network (PyTorch): ~90%
- Thinking of ideas for the new enhanced system:
  - Ensemble classifier that learns weights of fine-tuned BERT & feature vector
    - Need to brainstorm what to include in the feature vector:
      - Punctuation
      - Repeated letters
      - (Something with wordnet? Semantic/syntactice parse?)

### 4-21 Hilly

#### Stuff that needs to be done on classifier:
- Tune hyperparamers
- Test on dev set
- Decide whether to add additional layers
- Print outputs to pretty graphs

#### Training RNN 
- Training with 768 hidden layers, BCELoss(), LR=0.005, epochs=5, results in the loss functions going down, but accuracy staying about the same. Getting accuracy scores hovering around 60%.

### 4-11 Hilly
- Read the ACL paper on shared task--they used BERT as a baseline.
- Found out that [TensorFlow](https://www.tensorflow.org/text/tutorials/classify_text_with_bert) and [PyTorch](https://pytorch.org/hub/huggingface_pytorch-transformers/) have pre-built BERT models.

### 4-7 Hilly 

- Created lab notebook and created a data folder in the src directory.
- Tagged [deliverable 1](https://github.com/hasteinmetz/LING573-Project/releases/tag/D1).
