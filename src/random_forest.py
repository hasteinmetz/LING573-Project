import json
import time
import utils
import argparse
import pandas as pd
from typing import *
from featurizer import featurize
from feature_selection import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

def train_random_forest(config: dict, lexical_features, labels) -> RandomizedSearchCV:
	'''
	arguments:
		- config: grid of parameter options to configure rf model with 
		- lexical_features: array of lexical features, one row per sample in data
		- labels: corresponding classification label for each sample
	returns:
		sklearn.ensemble.RandomForestClassifier
	initializes a hyperparameter tuning scheme based off of parameter options provided by config file
	and finds the optimal set of parameters for the best-performing random forest classifier.

	uses cross-fold validation while training.
	'''
	rf = RandomForestClassifier()
	hyperparam_tuner = RandomizedSearchCV(estimator=rf, param_distributions=config["param_grid"], n_iter=config["n_iter"], \
		cv=config["cv"], verbose=config["verbose"], random_state=config["random_state"], n_jobs=config["n_jobs"])
	hyperparam_tuner.fit(lexical_features, labels)
	return hyperparam_tuner

def main(args: argparse.Namespace) -> None:
	start_time = time.time()

	print("loading data...")
	train_sentences, train_labels = utils.read_data_from_file(args.train_data_path)
	dev_sentences, dev_labels = utils.read_data_from_file(args.dev_data_path)

	print("preparing hurtlex dictionary...")
	hurtlex_dict, hurtlex_feat_list = utils.read_from_tsv(args.hurtlex_path)
	print("featurizing training and dev data...")
	train_feat_vector = featurize(train_sentences, train_labels, hurtlex_dict, hurtlex_feat_list)
	dev_feat_vector = featurize(dev_sentences, dev_labels, hurtlex_dict, hurtlex_feat_list)
	print("reducing feature dimensions...")
	train_feature_vector, feat_indices = k_perc_best_f(train_feat_vector, train_labels, 70)
	#train_feature_vector, feat_indices = k_best_f(train_feat_vector, train_labels, 40)
	dev_feature_vector = prune_test(dev_feat_vector, feat_indices)

	print("finding optimal parameter settings...")
	print(f"({utils.get_time(start_time)}) starting random forest hyperparameter search...\n")
	training_config = utils.load_json_config(args.rf_train_config)
	rf_trainer = train_random_forest(training_config, train_feature_vector, train_labels)
	print(f"({utils.get_time(start_time)}) training complete!")

	print("outputting optimal parameters...")
	optimal_params = rf_trainer.best_params_
	with open(args.param_output_path, 'w', encoding='utf-8') as f:
		json.dump(optimal_params, f)
	
	print("best score on training data: {}".format(rf_trainer.best_score_))
	rf_classifier = rf_trainer.best_estimator_
	dev_acc = rf_classifier.score(dev_feature_vector, dev_labels)
	print("best score on dev data:{}".format(dev_acc))
	dev_pred = rf_classifier.predict(dev_feature_vector)
	dev_out_d = {'sentence': dev_sentences, 'predicted': dev_pred, 'correct_label': dev_labels}
	dev_out = pd.DataFrame(dev_out_d)
	dev_out.to_csv(args.results_output_path, index=False, encoding='utf-8')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--rf_train_config", help="configuration settings for random forest classifier")
	parser.add_argument("--train_data_path", help="path to input training data file")
	parser.add_argument("--dev_data_path", help="path to input dev data file")
	parser.add_argument("--hurtlex_path", help="path to hurtlex lexicon file")
	parser.add_argument("--results_output_path", help="path to where classification results of best model should be written to")
	parser.add_argument("--param_output_path", help="path to where optimal parameters should be written to")
	args = parser.parse_args()

	main(args)
