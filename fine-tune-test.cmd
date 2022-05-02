executable 		            = src/fine-tune.sh 
getenv     		            = true
arguments		            = 
transfer_output_files       = outputs/test/fine-tune_results.txt
output			            = outputs/test/ft.out
error      		            = outputs/test/ft.error
log        		            = outputs/test/ft.log
request_GPUs                = 1
transfer_executable 	    = false
request_memory 		        = 2*1024
notification                = complete
queue
