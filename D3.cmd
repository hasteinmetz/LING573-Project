executable           = src/run_ensemble.sh 
getenv               = true
arguments            = 
transfer_output_files = outputs/D3/ensemble/d3_out.txt, log
output               = outputs/D3/ensemble/condor.out
error                = outputs/D3/ensemble/condor.error
log                  = outputs/D3/ensemble/condor.log
request_GPUs         = 1
transfer_executable  = false
request_memory       = 2*1024
queue
