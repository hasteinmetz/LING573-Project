#!/usr/bin/env python

'''References:
- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://huggingface.co/transformers/v3.2.0/custom_datasets.html#seq-imdb
'''

import time
import torch
import utils
import argparse
import pandas as pd
import numpy as np
from typing import *
from datasets import load_metric
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForSequenceClassification as RobertaModel
from transformers import RobertaTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, EvalPrediction


class FineTuneDataSet(Dataset):
    '''Class creates a list of dicts of sentences and labels
    and behaves like a list but also stores sentences and labels for
    future use'''
    def __init__(self, sentences: List[str], labels: List[int]):
        self.sentences = sentences
        self.labels = labels

    def tokenize_data(self, tokenizer: RobertaTokenizer):
        if not hasattr(self, 'encodings'):
            # encode the data
            self.encodings = tokenizer(self.sentences, return_tensors="pt", padding=True)
            self.input_ids = self.encodings['input_ids']

    def __getitem__(self, index: int,):
        if not hasattr(self, 'encodings'):
            raise AttributeError("Did not initialize encodings or input_ids")
        else:
            item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
            item['label'] = torch.tensor(self.labels[index])
            return item

    def __len__(self):
        return len(self.labels)


def get_time(start_time: float) -> str:
    minutes, sec = divmod(time.time() - start_time, 60)
    return f"{str(round(minutes))}min {str(round(sec))}sec"


def metrics(measure, evalpred: EvalPrediction) -> tuple:
    '''Helper function to compute the f1 and accuracy scores using
    the Transformers package's data structures'''
    logits, labels = evalpred
    predictions = np.argmax(logits, axis=-1)
    return measure.compute(predictions=predictions, references=labels)


def evaluate(model: RobertaModel, batch_size: int, 
    test_data: FineTuneDataSet, measures: List[str], device: str) -> None:
    '''Evaluate model performance on the test texts'''
    # set the model to eval mode
    model.eval()
    model.to(device)

    # create a list of metrics to store data
    metrics = []
    for metric in measures:
        m = load_metric(metric)
        metrics.append(m)

    # convert dataset to a pytorch format and batch the data
    eval_dataloader = DataLoader(test_data, batch_size=batch_size)

    # store the argmax of each batch
    predictions = []

    # iterate through batches to get outputs
    for batch in eval_dataloader:
        batch['labels'] = batch.pop('label')
        labels = batch['labels']

        # assign each element of the batch to the device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        # get batched results
        logits = outputs.logits

        # add batch to output
        pred_argmax = torch.argmax(logits, dim = -1)
        as_list = pred_argmax.clone().detach().to('cpu').tolist()
        predictions.append(as_list)

        # add batched results to metrics
        for m in metrics:
            m.add_batch(predictions=pred_argmax, references=labels)
    
    # output metrics to standard output
    values = f"" # empty string 
    for m in metrics:
        val = m.compute()
        values += f"{m.name}:\n\t {val}\n"
    print(values)
    return np.concatenate(predictions)


def train_new_model(args: argparse.Namespace, train_data: FineTuneDataSet, dev_data: FineTuneDataSet, 
                    data_collator: DataCollatorWithPadding, tokenizer: RobertaTokenizer, 
                    comp_measure: Callable, start_time: float) -> RobertaModel:
    '''**Fine-tune the RoBERTa model using input data**
        Args:
            - args: arguments passed into the program
                - learning_rate: learning rate of model
                - batch_size: batch size to train model (keep this below 50)
                - epochs: number of times to iterate over the data
            - train_data: the training data to be used
            - dev_data: the dataset to evaluate on
            - tokenizer: the tokenizer to be used to encode sentences
            - comp_measure: the measurement you want to report at each :evaluation_strategy:
            - start_time: the time the program began running
    '''

    # initialize model
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # set the arguments
    fine_tune_args = TrainingArguments(
        output_dir = './outputs/test/',
        learning_rate = args.learning_rate,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )

    # create a trainer
    print(f"({utils.get_time(start_time)}) Fine-tuning the model...\n")
    fine_tuned_model = Trainer(
        model=roberta_model,
        args=fine_tune_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_measure,
    )

    #fine-tune the model
    fine_tuned_model.train()
    
    return fine_tuned_model.model

def roberta_io(model: RobertaModel, sentences: List[str],
    batch_size: int, device: str) -> np.ndarray:
    '''Function to export to other scripts that takes a model and list of sentences
    and outputs an ndarray of predicted classes. It also takes a batch_size (for evaluation)
    and a device (cpu or cuda)'''
    # initialize roberta tokenizer and pretrained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # read in the training and development data
    train_sentences, train_labels = utils.read_data_from_file(sentences)
    train_data = FineTuneDataSet(train_sentences, train_labels)

    # evaluate model
    y_pred_train = evaluate(model, batch_size, train_data, ['f1', 'accuracy'], device)

    return y_pred_train

def main(args: argparse.Namespace) -> None:
    # get starting time
    start_time = time.time()

    # check if cuda is avaiable
    if torch.cuda.is_available():
        device = "cuda"
        torch.device(device)
        print(f"({utils.get_time(start_time)}) Using {device} device")
        print(f"Using the GPU:{torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        torch.device(device)
        print(f"({utils.get_time(start_time)}) Using {device} device")

    print(f"({utils.get_time(start_time)}) Reading data in from files...\n")
    # initialize roberta tokenizer and pretrained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # read in the training and development data
    train_sentences, train_labels = utils.read_data_from_file(args.train_sentences)
    dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences)

    # change the dimensions of the input sentences only when debugging (adding argument --debug 1)
    if args.debug == 1:
        np.random.shuffle(train_sentences)
        np.random.shuffle(train_labels)
        train_sentences, train_labels = train_sentences[0:50], train_labels[0:50]
    #if args.debug == 1:
        np.random.shuffle(dev_sentences)
        np.random.shuffle(dev_labels)
        dev_sentences, dev_labels = dev_sentences[0:50], dev_labels[0:50]

    # load data into dataloader
    train_data = FineTuneDataSet(train_sentences, train_labels)
    dev_data = FineTuneDataSet(dev_sentences, dev_labels)

    # get roberta encodings for each sentence (see FineTuneDataSet class)
    train_data.tokenize_data(tokenizer)
    dev_data.tokenize_data(tokenizer)

    print(f"({utils.get_time(start_time)}) Initalizating RoBERTa and creating data collator...\n")

    # create a data collator to obtain the encoding (and padding) for each sentence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialize metrics
    accuracy = load_metric("accuracy")
    get_accuracy = lambda x: metrics(accuracy, x)

    # if there is no existing model, then train a new model
    if args.model_folder != 'None':
        try:
            roberta_model = RobertaModel.from_pretrained(args.model_folder)
            print(f"({utils.get_time(start_time)}) Using model saved to folder {args.model_folder}")
        except FileNotFoundError(args.model_folder) as err:
            print(f"({utils.get_time(start_time)}) Could not get model from folder {args.model_folder}...")
    else:
        roberta_model = RobertaModelWrapper(args.batch_size, device, optimizer: AdamW, learning_rate: float)

    # evaluate the model's performance
    print(f"\n({utils.get_time(start_time)}) Evaluating the Transformer model on training data\n")
    y_pred_train = evaluate(roberta_model, args.batch_size, train_data, ['f1', 'accuracy'], device)

    print(f"\n({utils.get_time(start_time)}) Evaluating the Transformer model on dev data\n")
    y_pred_dev = evaluate(roberta_model, args.batch_size, dev_data, ['f1', 'accuracy'], device)

    #write results to output file
    train_out_d = {'sentence': train_data.sentences, 'predicted': y_pred_train, 'correct_label': train_data.labels}
    dev_out_d = {'sentence': dev_data.sentences, 'predicted': y_pred_dev, 'correct_label': dev_data.labels}
    train_out, dev_out = pd.DataFrame(train_out_d), pd.DataFrame(dev_out_d)
    dev_out.to_csv(args.output_file, index=False, encoding='utf-8')

    # write missing examples to one particular file
    df = pd.concat((train_out, dev_out), axis=0)

    # filter the data so that only negative examples are there
    data_filtered = df.loc[~(df['predicted'] == df['correct_label'])]
    data_filtered.to_csv('src/data/roberta-misclassified-examples.csv', index=False, encoding='utf-8')

    # save the model
    if args.save_file != 'None':
        roberta_model.save_pretrained(args.save_file)

    print(f"({utils.get_time(start_time)}) Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sentences', help="path to input training data file")
    parser.add_argument('--dev_sentences', help="path to input dev data file")
    parser.add_argument('--learning_rate', help="(float) learning rate of the classifier", type=float)
    parser.add_argument('--batch_size', help="(int) batch size of mini-batches for training", type=int)
    parser.add_argument('--epochs', help="(int) number of epochs for training", type=int)
    parser.add_argument('--debug', help="(1 or 0) train on a smaller training set for debugging", default=0, type=int)
    parser.add_argument('--output_file', help="path to output data file")
    parser.add_argument('--model_folder', help="path to load a pretrained model from a folder", default='None', type=str)
    parser.add_argument('--save_file', help="path to save the pretrained model", default='None', type=str)
    args = parser.parse_args()

    main(args)
