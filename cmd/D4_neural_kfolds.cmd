executable            = src/executables/d4_neural_kfolds.sh 
getenv                = true
output                = outputs/D4/nn_ensemble_kfolds/$(job)/condor-$(dim_red).out
error                 = outputs/D4/nn_ensemble_kfolds/$(job)/condor-$(dim_red).error
log                   = outputs/D4/nn_ensemble_kfolds/$(job)/condor-$(dim_red).log
request_GPUs          = 1
requirements          = (CUDACapability >= 10.2) && $(requirements:True)
Rank                  = (machine == "patas-gn2.ling.washington.edu") || (machine == "patas-gn1.ling.washington.edu") || (machine == "patas-gn3.ling.washington.edu")
transfer_executable   = false
request_memory        = 3*1024
stream_output         = True
queue job, dim_red, arguments from (
    humor, pca, "--index 1 --job $(job) --dim_reduc_method $(dim_red) --test 1"
    controversy, pca, "--index 2 --job $(job) --dim_reduc_method $(dim_red) --test 1"
    humor, kbest, "--index 1 --job $(job) --dim_reduc_method $(dim_red) --test 1"
    controversy, kbest, "--index 2 --job $(job) --dim_reduc_method $(dim_red) --test 1"
)
