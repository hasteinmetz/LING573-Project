import nltk
import csv
import numpy as np

# call with get_ner_matrix(path/to/input_file)


def get_ner_matrix(input_file):  # returns a numpy array of [num_documents, 6] where 6 is the number of NER categories
    input_file = input_file
    label_list = ["GPE", "PERSON", "ORGANIZATION", "FACILITY", "LOCATION", "GSP"]  # hardcoded to ensure length
    label_dict = {
        "GPE": 0,
        "PERSON": 1,
        "ORGANIZATION": 2,
        "FACILITY": 3,
        "LOCATION": 4,
        "GSP": 5
    }
    total_list = []
    with open(input_file, encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
            tokenized = nltk.word_tokenize(row[0])  # tokenize
            tagged = nltk.pos_tag(tokenized)  # get pos-tags, necessary for nltk.ne_chunk
            ne_tree = nltk.ne_chunk(tagged)  # develop tree, which defines what is a named entity

            ne_list = []

            for item in ne_tree:
                if not isinstance(item, tuple):  # non-Named-entities are saved as tuples, others are Named entities
                    ne_list.append(item)  # append Named entities

            zerod = np.zeros(len(label_list))
            for item in ne_list:
                label = item.label()
                index = label_dict.get(label)
                if index is not None:
                    zerod[index] += 1
            total_list.append(zerod)

    ner_array = np.asarray(total_list)
    csv_file.close()

    return ner_array
