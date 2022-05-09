executable           = src/run_ensemble.sh 
getenv               = true
arguments            = 
output               = outputs/condor.out
error                = outputs/condor.error
log                  = outputs/condor.log
request_GPUs         = 1
transfer_executable  = false
request_memory       = 2*1024
queue
