executable 		            = src/fine-tune.sh 
getenv     		            = true
# type "--debug 1" in arguments to run the model on fewer training and testing samples
# type "--model_folder {path}" to load the model at the path specified
arguments		            = 
transfer_output_files       = outputs/test/fine-tune_results.csv, src/data/roberta-misclassified-examples.csv, src/models/roberta-fine-tuned
output			            = outputs/test/ft.out
error      		            = outputs/test/ft.error
log        		            = outputs/test/ft.log
request_GPUs                = 1
transfer_executable 	    = false
request_memory 		        = 2*1024
notification                = complete
queue
