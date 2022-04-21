Preprocessing done through 'preprocess.py'.
Generates preprocessed data from 'hahackathon_train.csv', or from whatever csv file set as input_training_data.
Parameters used to generate 'hahackathon_prepo2_train/test/dev.csv':
    Seed: 573
    Training Ratio: 0.8
    No preprocessing done of the actual text, just to remove unwanted data. Only left with the joke and the label.
    Data is now split in 80/10/10. To change this ratio, change the "training ratio" variable.

If you want to easily preprocess data from another csv file, you may have to alter the lines that actually write the
rows to the csv file (as of upload, 87, 89, 96, and 98.). These lines assume that the wanted data is in the second and
third columns of the .csv, as shown by [1:3].