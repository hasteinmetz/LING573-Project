executable 		  = src/run_adam.sh 
getenv     		  = true
arguments		  = 
transfer_output_files     = outputs/d2/d2_adam_out.txt, adam_log
output			  = outputs/condor_adam.out
error      		  = outputs/condor_adam.error
log        		  = outputs/condor_adam.log
request_GPUs              = 1
transfer_executable 	  = false
request_memory 		  = 2*1024
queue
