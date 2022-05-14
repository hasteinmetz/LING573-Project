executable           = ../src/executables/d4_rfens_pca.sh 
getenv               = true
arguments            = 
output               = outputs/D4/ensemble-test/condor.out
error                = outputs/D4/ensemble-test/condor.error
log                  = outputs/D4/ensemble-test/condor.log
request_GPUs         = 1
transfer_executable  = false
request_memory       = 3*1024
queue
