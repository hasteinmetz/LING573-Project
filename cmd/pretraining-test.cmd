executable            = src/executables/pretraining.sh 
getenv                = true
output                = outputs/D4/pretraining-test/condor-$(job).out
error                 = outputs/D4/pretraining-test/condor-$(job).error
log                   = outputs/D4/pretraining-test/condor-$(job).log
request_GPUs          = 1
requirements          = (CUDACapability >= 10.2) && $(requirements:True)
Rank                  = (machine == "patas-gn2.ling.washington.edu") || (machine == "patas-gn1.ling.washington.edu") || (machine == "patas-gn3.ling.washington.edu")
transfer_executable   = false
request_memory        = 2*1024
queue job, arguments from (
    regression, "--job regression --train_sentences src/data/D4_hahackathon_prepo1_train.csv --dev_sentences src/data/D4_hahackathon_prepo1_dev.csv --pretrain_data src/data/D4_1_hahackathon_prepo1_train.csv"
)