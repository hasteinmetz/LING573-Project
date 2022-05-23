#!/usr/bin/env python

'''References:
- https://huggingface.co/docs/transformers/tasks/sequence_classification
- https://huggingface.co/transformers/v3.2.0/custom_datasets.html#seq-imdb
'''

import time
import torch
import utils
import argparse
import json
import numpy as np
import pandas as pd
from typing import *
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, PreTrainedModel, RobertaConfig
from finetune_dataset import FineTuneDataSet
from transformers import RobertaModel as RobertaLM
from transformers import RobertaForSequenceClassification as RobertaSeqCls
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer, EvalPrediction
from sklearn.utils import shuffle
import sys

def metrics(measure, evalpred: EvalPrediction) -> tuple:
    '''Helper function to compute the f1 and accuracy scores using
    the Transformers package's data structures'''
    logits, labels = evalpred
    predictions = np.argmax(logits, axis=-1)
    return measure.compute(predictions=predictions, references=labels)


def evaluate(model: RobertaSeqCls, batch_size: int, 
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

# TODO: PRETRAIN ON REGRESSION
# CAN SIMPLY DO model.roberta = roberta

def pretrain_model(args: dict, data: FineTuneDataSet, data_collator: DataCollatorWithPadding, 
                    tokenizer: RobertaTokenizer, comp_measure: Callable, start_time: float) -> PreTrainedModel:
    '''Pretrain a model on headlines from the onion. Use a classification task'''
    # initialize sequence classifier
    
    config = RobertaConfig(name_or_path='roberta-base', problem_type='regression')
    seq_classifier_model = RobertaSeqCls(config)

    # set the arguments
    pretrain_tune_args = TrainingArguments(
        output_dir = './outputs/test/',
        learning_rate = args['learning_rate'],
        per_device_train_batch_size = args['batch_size'],
        num_train_epochs=args['epochs'],
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )

    # create a trainer
    print(f"({utils.get_time(start_time)}) Pre-training the model...")
    pretrain_tuned_model = Trainer(
        model=seq_classifier_model,
        args=pretrain_tune_args,
        train_dataset=data,
        eval_dataset=data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=comp_measure,
    )

    pretrain_tuned_model.train()

    return pretrain_tuned_model.model


def fine_tune_model(model: RobertaSeqCls, args: argparse.Namespace,
                    train_data: FineTuneDataSet, dev_data: FineTuneDataSet, 
                    data_collator: DataCollatorWithPadding, tokenizer: RobertaTokenizer, 
                    comp_measure: Callable, start_time: float) -> RobertaSeqCls:
    '''**Fine-tune the RoBERTa model using input data**
        Args:
            - args: arguments passed into the program
                - learning_rate: learning rate of model
                - batch_size: batch size to train model (keep this below 50)
                - epochs: number of times to iterate over the data
            - model: the model desired to fine-tune (could be pre-trained)
            - train_data: the training data to be used
            - dev_data: the dataset to evaluate on
            - tokenizer: the tokenizer to be used to encode sentences
            - comp_measure: the measurement you want to report at each :evaluation_strategy:
            - start_time: the time the program began running
    '''

    # initialize sequence classifier
    seq_classifier_model = RobertaSeqCls.from_pretrained('roberta-base', problem_type='single_label_classification')
    seq_classifier_model.roberta = model.roberta

    # set the arguments
    fine_tune_args = TrainingArguments(
        output_dir = './outputs/test/',
        learning_rate = args['learning_rate'],
        per_device_train_batch_size = args['batch_size'],
        per_device_eval_batch_size = args['batch_size'],
        num_train_epochs=args['epochs'],
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )

    # create a trainer
    print(f"({utils.get_time(start_time)}) Fine-tuning the model...")
    fine_tuned_model = Trainer(
        model=seq_classifier_model,
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

# def roberta_io(model: RobertaSeqCls, sentences: List[str],
#     batch_size: int, device: str) -> np.ndarray:
#     '''Function to export to other scripts that takes a model and list of sentences
#     and outputs an ndarray of predicted classes. It also takes a batch_size (for evaluation)
#     and a device (cpu or cuda)'''
#     # initialize roberta tokenizer and pretrained model
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#     # read in the training and development data
#     train_sentences, train_labels = utils.read_data_from_file(sentences)
#     train_data = FineTuneDataSet(train_sentences, train_labels)

#     # evaluate model
#     y_pred_train = evaluate(model, batch_size, train_data, ['f1', 'accuracy'], device)

#     return y_pred_train

def main(args: argparse.Namespace, pretrain_args: dict, finetune_args: dict) -> None:
    # get starting time
    start_time = time.time()

    # check if cuda is avaiable
    if torch.cuda.is_available():
        device = "cuda"
        torch.device(device)
        print(f"({utils.get_time(start_time)}) Using {device} device", file=sys.stderr)
        print(f"Using the GPU:{torch.cuda.get_device_name(0)}", file=sys.stderr)
    else:
        device = "cpu"
        torch.device(device)
        print(f"({utils.get_time(start_time)}) Using {device} device", file=sys.stderr)

    print(f"({utils.get_time(start_time)}) Reading data in from files...\n", file=sys.stderr)
    # initialize roberta tokenizer and pretrained model
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # read in pretraining data
    pretrain_sents, pretrain_labels = utils.read_data_from_file(args.pretrain_data, index=2)
    print(pretrain_sents[0:2], pretrain_labels.shape)

    # read in the training and development data
    ind = 1 if args.job == 'humor' else 2
    train_sentences, train_labels = utils.read_data_from_file(args.train_sentences, index=ind)
    dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences, index=ind)

    # change the dimensions of the input sentences only when debugging (adding argument --debug 1)
    if args.debug == 1:
        shuffled_tr_sentences, shuffled_tr_labels = shuffle(train_sentences, train_labels, random_state = 0)
        train_sentences, train_labels = shuffled_tr_sentences[0:50], shuffled_tr_labels[0:50]
        shuffled_te_sentences, shuffled_te_labels = shuffle(dev_sentences, dev_labels, random_state = 0)
        dev_sentences, dev_labels = shuffled_te_sentences[0:50], shuffled_te_labels[0:50]
        shuffled_pt_sentences, shuffled_pt_labels = shuffle(train_sentences, train_labels, random_state = 0)
        pretrain_sents, pretrain_labels = shuffled_pt_sentences[0:50], shuffled_pt_labels[0:50]

    # load data into dataloader
    pretrain_data = FineTuneDataSet(pretrain_sents, pretrain_labels)
    train_data = FineTuneDataSet(train_sentences, train_labels)
    dev_data = FineTuneDataSet(dev_sentences, dev_labels)

    # get roberta encodings for each sentence (see FineTuneDataSet class)
    pretrain_data.tokenize_data(tokenizer)
    train_data.tokenize_data(tokenizer)
    dev_data.tokenize_data(tokenizer)

    print(f"({utils.get_time(start_time)}) Initalizating RoBERTa and creating data collator...\n", file=sys.stderr)

    # create a data collator to obtain the encoding (and padding) for each sentence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialize metrics
    accuracy = load_metric("accuracy")
    get_accuracy = lambda x: metrics(accuracy, x)

    # pretrain the model on other data
    print(f"({utils.get_time(start_time)}) Pre-train the model on other data...\n", file=sys.stderr)
    pretrained_model = pretrain_model(pretrain_args, pretrain_data, data_collator, tokenizer, get_accuracy, start_time)
    
    # train the model on the training data
    print(f"({utils.get_time(start_time)}) Fine-tune the model on the training data...\n", file=sys.stderr)
    roberta_model = fine_tune_model(pretrained_model, finetune_args, train_data, dev_data, 
        data_collator, tokenizer, get_accuracy, start_time)

    # evaluate the model's performance
    print(f"\n({utils.get_time(start_time)}) Evaluating the Transformer model on training data\n", file=sys.stderr)
    y_pred_train = evaluate(roberta_model, pretrain_args['batch_size'], train_data, ['f1', 'accuracy'], device)

    print(f"\n({utils.get_time(start_time)}) Evaluating the Transformer model on dev data\n", file=sys.stderr)
    y_pred_dev = evaluate(roberta_model, finetune_args['batch_size'], dev_data, ['f1', 'accuracy'], device)

    #write results to output file
    train_out_d = {'sentence': train_data.sentences, 'predicted': y_pred_train, 'correct_label': train_data.labels}
    dev_out_d = {'sentence': dev_data.sentences, 'predicted': y_pred_dev, 'correct_label': dev_data.labels}
    train_out, dev_out = pd.DataFrame(train_out_d), pd.DataFrame(dev_out_d)
    dev_out.to_csv(finetune_args['output_path'] + "-" + args.job, index=False, encoding='utf-8')

    # write missing examples to one particular file
    df = pd.concat((train_out, dev_out), axis=0)

    # filter the data so that only negative examples are there
    data_filtered = df.loc[~(df['predicted'] == df['correct_label'])]
    data_filtered.to_csv('src/data/roberta-misclassified-examples' + "-" + args.job + '.csv', index=False, encoding='utf-8')

    # save the model
    if finetune_args['save_model'] != 'None':
        roberta_model.save_pretrained(finetune_args['save_model'] + "-" + args.job)

    print(f"({utils.get_time(start_time)}) Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_sentences', help="path to input data to pretrain on")
    parser.add_argument('--pretrain_data', help="path to input training data file")
    parser.add_argument('--dev_sentences', help="path to input dev data file")
    parser.add_argument('--model_folder', help="path to a saved file to load")
    parser.add_argument('--debug', help="(1 or 0) train on a smaller training set for debugging", default=0, type=int)
    parser.add_argument('--job', help="to help name files when running batches", default='test', type=str)
    args = parser.parse_args()
    with open('src/configs/pretraining/pretrain.json', 'r') as f1:
        configs1 = f1.read()
        pretrain_args = json.loads(configs1)
    with open('src/configs/pretraining/finetune.json', 'r') as f2:
        configs2 = f2.read()
        finetune_args = json.loads(configs2)

    main(args, pretrain_args, finetune_args)
