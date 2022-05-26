executable           = ../src/executables/d4_rfens_pca_$(combo).sh 
getenv               = true
arguments            = 
output               = ../outputs/D4/rf_ensemble/condor_$(combo).out
error                = ../outputs/D4/rf_ensemble/condor_$(combo).error
log                  = ../outputs/D4/rf_ensemble/condor_$(combo).log
request_GPUs         = 1
request_memory       = 2*1024
queue combo from (
	pri_train
)
