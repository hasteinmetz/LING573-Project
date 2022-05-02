executable 		  = src/run.sh 
getenv     		  = true
arguments		  = 
transfer_output_files     = outputs/d2/d2_out.txt
output			  = outputs/condor.out
error      		  = outputs/condor.error
log        		  = outputs/condor.log
request_GPUs              = 1
transfer_executable 	  = false
request_memory 		  = 2*1024
queue
