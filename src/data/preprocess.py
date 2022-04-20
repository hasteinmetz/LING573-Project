"""
Eli Bales
"""
import csv
import random


def get_dif(list_1, list_2):  # gets all indices not used in a list of indices
    return set(list_1)-set(list_2)


def get_indices(a_list, train_size, rows):
    positive_ratio = len(a_list) / len(rows)  # gets the ratio of humorous to non-humorous jokes
    positive_train = int(train_size * positive_ratio)  # get number of humorous/non-humorous jokes we want for training

    train_indices = random.sample(range(len(a_list)), positive_train)  # gets training indices
    total_range = list(range(len(a_list)))
    test_indices = list(get_dif(total_range, train_indices))  # gets testing indices
    return train_indices, test_indices


def create_dev_file(positive_list, negative_list, dev_size, output_file):
    with open(output_file, "w", newline='', encoding="utf-8") as dev_csv:
        my_writer = csv.writer(dev_csv)

        dev_size = int(dev_size/2)
        rand_pos_idx = random.sample(range(len(positive_list)), dev_size)
        rand_neg_idx = random.sample(range(len(negative_list)), dev_size)

        for i in rand_pos_idx:
            my_writer.writerow(positive_list[i][1:3])
        for j in rand_neg_idx:
            my_writer.writerow(negative_list[j][1:3])
    dev_csv.close()


def main():
    # ---------------- PARAMETERS ----------------
    input_train_data = "hahackathon_train.csv"
    output_train_data = "hahackathon_prepo1_train.csv"
    output_test_data = "hahackathon_prepo1_test.csv"
    output_dev_data = "hahackathon_prepo1_dev.csv"
    create_dev = False
    dev_size = 100
    random.seed(573)
    training_ratio = 0.9
    # -------------------------------------------

    fields = []
    rows = []
    positive_list = []
    negative_list = []

    with open(input_train_data, 'r', encoding="utf-8") as csv_file:  # open .csv file
        my_reader = csv.reader(csv_file)

        fields = next(my_reader)

        for row in my_reader:  # save all the rows
            rows.append(row)

    csv_file.close()

    for document in rows:  # for every document
        if document[2] == "1":  # if the 'is_humorous' label is TRUE
            positive_list.append(document)  # add to the positive list
        else:
            negative_list.append(document)  # add to the negative list

    # print(len(positive_list))  # 4932
    num_documents = len(rows)  # 8000 in hahackathon

    train_size = int(num_documents * training_ratio)  # 7200 = 8000 * 0.9
    train_pos_idx, test_pos_idx = get_indices(positive_list, train_size, rows)  # get humorous indices
    train_neg_idx, test_neg_idx = get_indices(negative_list, train_size, rows)  # get non-humorous indices

    train_pos_idx.sort()  # sort for easier viewing, not necessary
    test_pos_idx.sort()

    train_neg_idx.sort()  # sort for easier viewing, not necessary
    test_neg_idx.sort()

    with open(output_train_data, "w", newline='', encoding="utf-8") as train_csv:  # create training data file
        my_writer = csv.writer(train_csv)

        for i in train_pos_idx:  # for all training positive indices
            my_writer.writerow(positive_list[i][1:3])
        for j in train_neg_idx:  # for all training negative indices
            my_writer.writerow(negative_list[j][1:3])
    train_csv.close()

    with open(output_test_data, "w", newline='', encoding="utf-8") as test_csv:  # create testing data file
        my_writer2 = csv.writer(test_csv)

        for i in test_pos_idx:  # for all testing positive indices
            my_writer2.writerow(positive_list[i][1:3])
        for j in test_neg_idx:  # for all testing negative indices
            my_writer2.writerow(negative_list[j][1:3])
    test_csv.close()

    if create_dev:
        create_dev_file(positive_list, negative_list, dev_size, output_dev_data)


main()
