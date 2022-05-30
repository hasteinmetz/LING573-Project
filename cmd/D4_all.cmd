executable            = src/executables/d4_neural_kfolds.sh 
getenv                = true
output                = outputs/D4/$(job)/D4_scores.out
error                 = outputs/D4/$(job)/D4.error
log                   = outputs/D4/$(job)/D4.log
request_GPUs          = 1
requirements          = (CUDACapability >= 10.2) && $(requirements:True)
transfer_executable   = false
request_memory        = 3*1024
stream_output         = True
queue job, dim_red, arguments from (
    primary, kbest, "--debug 1 --index 1 --job $(job) --dim_reduc_method $(dim_red) --test eval"
    adaptation, kbest, "--debug 1 --index 2 --job $(job) --dim_reduc_method $(dim_red) --test eval"
)
