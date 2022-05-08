executable 		            = src/fine-tune-$(model).sh 
getenv     		            = true
# type "--debug 1" in arguments to run the model on fewer training and testing samples
# type "--model_folder {path}" to load the model at the path specified
arguments		            = 
transfer_output_files       = outputs/D3/$(model)/fine-tune_results.csv, src/data/$(model)-misclassified-examples.csv, src/models/$(model)-fine-tuned
output			            = outputs/D3/$(model)/ft.out
error      		            = outputs/D3/$(model)/ft.error
log        		            = outputs/D3/$(model)/ft.log
request_GPUs                = 1
transfer_executable 	    = false
request_memory 		        = 2*1024
notification                = complete
queue model from batch_file.txt
