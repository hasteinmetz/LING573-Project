# 573 Project on Humor

Eli, Avani, Pangbo, and Hilly's project for LING 573 at the University of Washington.

src/ contains the source code and data used for this project.

results/ contains the results of our system.

doc/ contains documentation of our system.

outputs/ contains the system outputs.

# D3 Instructions

1. Issue the following command from the root of this repo: condor_submit src/D3.cmd

The accuracy and F1 score of our finet-uned RoBERTa model will be written to the following file: src/outputs/D3/roberta/ft.out

This can be compared to using a fine-tuned BERTweet, whose output file is: src/outputs/D3/bertweet/ft.out

Our model performs a binary classification task on data from a joke corpus. The classification outputs will be written to src/outputs/D3/roberta/fine-tune_results.csv

RoBERTa results:

f1:

	 {'f1': 0.937178166838311}
	 
accuracy:

	 {'accuracy': 0.92375}
   
BERTweet

f1:

   {'f1': 0.9504550050556119}
   
accuracy:

   {'accuracy': 0.93875}


Our code requires the appropriate conda environment to run properly. We activate the environment, which is saved in our shared space on patas (/projects/assigned/2122_ling573_elibales), within either src/fine-tune-$(model).sh scripts itself.
