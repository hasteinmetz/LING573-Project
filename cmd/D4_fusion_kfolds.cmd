executable            = src/executables/run_fusion_kfolds.sh 
getenv                = true
output                = outputs/D4/ensemble-kfolds/$(job)/condor.out
error                 = outputs/D4/ensemble-kfolds/$(job)/condor.error
log                   = outputs/D4/ensemble-kfolds/$(job)/condor.log
request_GPUs          = 1
requirements          = (CUDACapability >= 10.2) && $(requirements:True)
Rank                  = (machine == "patas-gn2.ling.washington.edu") || (machine == "patas-gn1.ling.washington.edu") || (machine == "patas-gn3.ling.washington.edu")
transfer_executable   = false
request_memory        = 2*1024
stream_output         = True
queue job, arguments from (
    humor, "--index 1 --job $(job)"
    controversy, "--index 2 --job $(job)"
)
