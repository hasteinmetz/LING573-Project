executable 		  = src/run.sh 
getenv     		  = true
arguments		  = 
transfer_output_files     = outputs/baseline_output.csv
output			  = outputs/condor.out
error      		  = outputs/condor.error
log        		  = outputs/condor.log
<<<<<<< HEAD
=======
request_GPUs              = 1
>>>>>>> main
transfer_executable 	  = false
request_memory 		  = 2*1024
queue
