#!/usr/bin/env python

'''References:
- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://huggingface.co/transformers/v3.2.0/custom_datasets.html#seq-imdb
'''

import torch
import utils
import argparse
import numpy as np
import time
from typing import *
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification as RobertaModel
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, EvalPrediction
from datasets import load_metric


class FineTuneDataSet(torch.utils.data.Dataset):
    '''Class creates a list of dicts of sentences and labels
    and behaves list a list but also stores sentences and labels for
    future use'''
    def __init__(self, sentences: List[str], labels: List[int]):
        self.sentences = sentences
        self.labels = labels

    def tokenize_data(self, tokenizer: RobertaTokenizer):
        if not hasattr(self, 'encodings'):
            # encode the data
            self.encodings = tokenizer(self.sentences, return_tensors="pt", padding=True)
            self.input_ids = self.encodings['input_ids']

    def __getitem__(self, index: int):
        if not hasattr(self, 'encodings'):
            raise AttributeError("Did not initialize encodings or input_ids")
        else:
            item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
            item['label'] = torch.tensor(self.labels[index])
            return item

    def __len__(self):
        return len(self.labels)


def get_time(start_time: float):
    minutes, sec = divmod(time.time() - start_time, 60)
    return f"{str(round(minutes))}min {str(round(sec))}sec"


def evaluate(trainer: Trainer, test_data: torch.utils.data.Dataset, metrics: List[Callable]):
    '''Evaluate model performance on the test texts'''
    # get test text predictions
    predictions = trainer.predict(test_data)
    pred_argmax = np.argmax(predictions.predictions, axis = -1)
    print(f"Predictions argmax: (size = {pred_argmax.shape}\n{pred_argmax}")
    print(f"Predictions label_ids: (size = {predictions.label_ids.shape}\n{predictions.label_ids}")
    values = f""
    for metric in metrics:
        evaluate = load_metric(metric)
        val = evaluate.compute(predictions=pred_argmax, references=predictions.label_ids)
        values += f"{metric}:\n\t {val}\n"
    print(values)
    return pred_argmax


def main(args: argparse.Namespace):
    # get starting time
    start_time = time.time()

    # check if cuda is avaiable
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"({get_time(start_time)}) Using {device} device")

    print(f"({get_time(start_time)}) Reading data in from files...\n")
    # initialize roberta tokenizer and pretrained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # read in the training and development data
    train_sentences, train_labels = utils.read_data_from_file(args.train_sentences)
    dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences)

    # change the dimensions of the input sentences only when debugging (--debug 1)
    if args.debug == 1:
        np.random.shuffle(train_sentences)
        np.random.shuffle(train_labels)
        train_sentences, train_labels = train_sentences[0:50], train_labels[0:50]
    if args.debug == 1:
        np.random.shuffle(dev_sentences)
        np.random.shuffle(dev_labels)
        dev_sentences, dev_labels = dev_sentences[0:50], dev_labels[0:50]

    # load data into dataloader
    train_data = FineTuneDataSet(train_sentences, train_labels)
    dev_data = FineTuneDataSet(dev_sentences, dev_labels)

    # get roberta encodings for each sentence
    train_data.tokenize_data(tokenizer)
    dev_data.tokenize_data(tokenizer)

    print(f"({get_time(start_time)}) Initalizating RoBERTa and creating data collator...\n")

    # create a data collator to obtain the encoding (and padding) for each sentence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialize metrics
    accuracy = load_metric("accuracy")
    def metrics(evalpred: EvalPrediction) -> tuple:
        '''Helper function to compute the f1 and accuracy scores using
        the Transformers package's data structures'''
        logits, labels = evalpred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

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

    # fine-tune the model
    print(f"({get_time(start_time)}) Fine-tuning the model...\n")
    trainer = Trainer(
        model=roberta_model,
        args=fine_tune_args,
        train_dataset=train_data,
        eval_dataset=dev_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
    )
    trainer.train()

    # evaluate the model's performance
    print(f"\n({get_time(start_time)}) Evaluating the Transformer model\n")
    y_pred = evaluate(trainer, dev_data, ['f1', 'accuracy'])

    #write results to output file
    utils.write_output_to_file(args.output_file, dev_data.sentences, y_pred, encoding='utf-8')
    print(f"({get_time(start_time)}) Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sentences', help="path to input training data file")
    parser.add_argument('--dev_sentences', help="path to input dev data file")
    parser.add_argument('--learning_rate', help="(float) learning rate of the classifier", type=float)
    parser.add_argument('--batch_size', help="(int) batch size of mini-batches for training", type=int)
    parser.add_argument('--epochs', help="(int) number of epochs for training", type=int)
    parser.add_argument('--debug', help="(1 or 0) train on a smaller training set for debugging", default=0, type=int)
    parser.add_argument('--output_file', help="path to output data file")
    args = parser.parse_args()

    main(args)
