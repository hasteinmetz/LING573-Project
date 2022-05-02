#!/usr/bin/env python

'''References:
- https://huggingface.co/docs/transformers/tasks/sequence_classification
'''

import torch
import utils
import argparse
import numpy as np
from typing import *
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification as RobertaModel
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from datasets import load_metric


def metrics(eval_batch):
    '''Evaluate a batch of data'''
    logits, labels = eval_batch
    predicted_labels = np.argmax(logits, axis=1)
    return (
        f1.compute(redictions=predicted_labels, references=labels),
        accuracy.compute(redictions=predicted_labels, references=labels),
    )


def evaluate(model, tokenizer, test_texts, test_labels):
    '''Evaluate model performance on the test texts'''
    inputs = tokenizer(test_texts, return_tensors="pt", padding=True)
    test_batched_data = torch.utils.data.Dataloader((test_texts, test_labels))

    model.eval()

    with torch.no_grad():
        logits = model(**test_texts).logits
        

def main(args: argparse.Namespace):
    # check if cuda is avaiable
	device = "cuda" if torch.cuda.is_available() else "cpu"
	print(f"Using {device} device")

    print("parsing data...")
    # read in the training and development data
	train_sentences, train_labels = utils.read_data_from_file(args.train_sentences)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_sentences)

    # initialize roberta tokenizer and pretrained model
	tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    # create a data collator to obtain the encoding (and padding) for each sentence
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # initialize metrics
    f1_score, accuracy = load_metric("f1"), load_metric("accuracy")

    # initialize model
    roberta_model = RobertaModel.from_pretrained('roberta-base')

    # set the arguments
    fine_tune_args = TrainingArguments(
        output_dir = './outputs',
        learning_rate = args.learning_rate
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch"
    )

    # fine-tune the model
    print("fine-tuning the model")
    trainer = Trainer(
        model=roberta_model,
        args=fine_tune_args,
        train_dataset=args.train_sentences,
        eval_dataset=args.dev_sentences,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics,
    )
    trainer.train()


