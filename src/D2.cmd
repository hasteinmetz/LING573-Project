executable 		  = run.sh 
getenv     		  = true
arguments		  = 
transfer_output_files     = outputs/baseline_output.csv
output			  = outputs/condor.out
error      		  = outputs/condor.error
log        		  = outputs/condor.log
request_GPUs              = 1
transfer_executable 	  = false
request_memory 		  = 2*1024
queue
