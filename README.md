# 573 Project on Humor

Eli, Avani, Pangbo, and Hilly's project for LING 573 at the University of Washington.

## D4 Instructions

1. Make sure you have Conda installed.
	- To install Conda, enter the following commands:
	- `wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86 64.sh`
	- `sh Anaconda3-2021.11-Linux-x86 64.sh`
2. Issue the following command from the root of this repo to fine-tune the models and print the results: condor_submit cmd/D4.cmd
	- The script will activate a Conda environment in the shared folder `/projects/assigned/2122_ling573_elibales/env/` 

The accuracy and F1 scores of our ensemble classification model can be found under: 
- src/results/D4/primary/evaltest/D4_scores.out
- src/results/D4/primary/devtest/D4_scores.out
- src/results/D4/adaptation/devtest/D4_scores.out
- src/results/D4/adaptation/devtest/D4_scores.out

### D4 results

- Primary:

	f1: {'f1': 0.9585439838220424}

	accuracy: {'accuracy': 0.9488139825218477}

- Adaptation:

	f1: {'f1': 0.5158730158730158}
	
	accuracy: {'accuracy': 0.5060728744939271}

The top 3 scores for the Hahackathon [shared task humor controversy subtask](https://competitions.codalab.org/competitions/27446#results) (our adaptation task) were:

1. accuracy: 0.5089 | f1: 0.6299
2. accuracy 0.4699 | f1: 0.6270
3. accuracy: 0.4553 | f1: 0.6249

## Directories

```
project
│
└─── src/ contains the source code and data used for this project.
│
└─── results/ contains the results of our system.
│
└─── doc/ contains documentation of our system.
│
│	│
│	└─── /archive contains unused files.
│	│
│	└─── /data contains the data used to train and evaluate the model.
│	│
│	└─── /configs contains JSON files used to configure the models.
│	│
│	└─── /exetuables contains bash scripts to run the models.
│
└─── outputs/ contains the system outputs.
│	│
│	└─── /Dx the results of each deliverable.
│
└─── cmd/ contains the HTCondor job submission files.
│
└─── bin/ a recycling bin for unused files.
```
