# 573 Project on Humor

Eli, Avani, Pangbo, and Hilly's project for LING 573 at the University of Washington.

src/ contains the source code and data used for this project.

results/ contains the results of our system.

doc/ contains documentation of our system.

outputs/ contains the system outputs.

# D2 Instructions

1. Issue the following command from the root of this repo: condor_submit src/D2.cmd

The accuracy and F1 score of our baseline model will be written to the following file: results/D2_scores.out
Our baseline model performs a binary classification task on data from a joke corpus. The classification outputs will be written to outputs/D2/d2_out.txt

Our code requires the appropriate conda environment to run properly. We activate the environment, which is saved in our shared space on patas (/projects/assigned/2122_ling573_elibales), within the run.sh script itself.
