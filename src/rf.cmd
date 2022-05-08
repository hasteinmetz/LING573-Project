executable           = src/train_rf.sh 
getenv               = true
arguments            = 
transfer_output_file = src/configs/random_forest.json
output               = outputs/rf_condor.out
error                = outputs/rf_condor.error
log                  = outputs/rf_condor.log
transfer_executable  = false
request_memory       = 2*1024
queue
