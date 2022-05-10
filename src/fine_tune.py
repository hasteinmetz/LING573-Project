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
from torch.optim import AdamW
from transformers import RobertaForSequenceClassification as RobertaModel
from transformers import RobertaTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, EvalPrediction, get_scheduler
from sklearn.metrics import accuracy_score

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

class RobertaModelWrapper:
    def __init__(self, batch_size: int, device: str, optimizer: torch.optim, learning_rate: float, model: RobertaModel = None):
    self.batch_size = batch_size
    if model:
        try:
            self.model = RobertaModel.from_pretrained(args.model_folder)
            print(f"({utils.get_time(start_time)}) Using model saved to folder {args.model_folder}")
        except FileNotFoundError(args.model_folder) as err:
            print(f"({utils.get_time(start_time)}) Could not get model from folder {args.model_folder}...")
    else:
        self.model = RobertaModel.from_pretrained('roberta-base')
    self.optim = optimizer(self.model.parameters(), lr=learning_rate)

    def train(self, train_data: FineTuneDataSet, device: str) -> RobertaModel:
        data_loader = DataLoader(train_data, self.batch_size)
        self.model.to(device)

        for batch in data_loader:
            batch['labels'] = batch.pop('label')
            labels = batch['labels']

            # assign each element of the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = self.model(**batch)

            logits = outputs.logits

            # get loss and update it
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log training progress
            pred_argmax = torch.argmax(logits, dim = -1)
            as_list = pred_argmax.clone().detach().to('cpu').tolist()

            print(f'accuracy: {accuracy_score(labels, as_list)}')
    
    def evaluate(self, test_data: FineTuneDataSet, measures: List[str], device: str) -> None:
        '''Evaluate model performance on the test texts'''
        # set the model to eval mode
        self.model.eval()
        self.model.to(device)

        # create a list of metrics to store data
        metrics = []
        for metric in measures:
            m = load_metric(metric)
            metrics.append(m)

        # convert dataset to a pytorch format and batch the data
        eval_dataloader = DataLoader(test_data, batch_size=self.batch_size)

        # store the argmax of each batch
        predictions = []

        # iterate through batches to get outputs
        for batch in eval_dataloader:
            batch['labels'] = batch.pop('label')
            labels = batch['labels']

            # assign each element of the batch to the device
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            
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


def metrics(measure, evalpred: EvalPrediction) -> tuple:
    '''Helper function to compute the f1 and accuracy scores using
    the Transformers package's data structures'''
    logits, labels = evalpred
    predictions = np.argmax(logits, axis=-1)
    return measure.compute(predictions=predictions, references=labels)


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

    # initialize metrics
    accuracy = load_metric("accuracy")
    get_accuracy = lambda x: metrics(accuracy, x)

    # if there is no existing model, then train a new model
    if args.model_folder != 'None':
        roberta_model = RobertaModelWrapper(args.batch_size, device, optimizer: AdamW, learning_rate: float, model: RobertaModel)
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