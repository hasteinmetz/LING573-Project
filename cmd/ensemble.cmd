executable           = src/executables/run_ensemble.sh 
getenv               = true
arguments            = 
output               = outputs/D4/ensemble-test/contro.out
error                = outputs/D4/ensemble-test/contro.error
log                  = outputs/D4/ensemble-test/contro.log
request_GPUs         = 1
requirements         = (CUDACapability >= 10.2) && $(requirements:True)
Rank                 = (machine == "patas-gn2.ling.washington.edu") || (machine == "patas-gn1.ling.washington.edu") || (machine == "patas-gn3.ling.washington.edu")
transfer_executable  = false
request_memory       = 2*1024
queue
