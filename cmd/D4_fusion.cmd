executable            = src/executables/run_fusion.sh 
getenv                = true
output                = outputs/D4/ensemble-test/$(job)/condor.out
error                 = outputs/D4/ensemble-test/$(job)/condor.error
log                   = outputs/D4/ensemble-test/$(job)/condor.log
request_GPUs          = 1
requirements          = (CUDACapability >= 10.2) && $(requirements:True)
Rank                  = (machine == "patas-gn2.ling.washington.edu") || (machine == "patas-gn1.ling.washington.edu") || (machine == "patas-gn3.ling.washington.edu")
transfer_executable   = false
request_memory        = 2*1024
queue job, arguments from (
    humor, "--error_path src/data/ensemble-misclassified-humor.csv --index 1 --model_save_location src/models/testing-ensemble/humor/ --output_file outputs/D4/ensemble-test/humor/ensemble-output.csv"
    controversy, "--error_path src/data/ensemble-misclassified-controvery.csv --index 2 --model_save_location src/models/testing-ensemble/controvery/ --output_file outputs/D4/ensemble-test/controversy/ensemble-output.csv"
)
