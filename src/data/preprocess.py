"""
Eli Bales
"""
import csv
import random


def get_dif(list_1, list_2):  # list_2 is a subset of list_1, finds all indices in list_1 not in list_2
    return set(list_1)-set(list_2)


def get_indices(a_list, train_size, rows):
    positive_ratio = len(a_list) / len(rows)  # gets the ratio of humorous to non-humorous jokes
    num_train = int(train_size * positive_ratio)  # get number of humorous/non-humorous jokes we want for training
    train_indices = random.sample(range(len(a_list)), num_train)  # gets training indices
    total_range = list(range(len(a_list)))
    test_indices = list(get_dif(total_range, train_indices))  # gets testing indices
    hold_size = len(test_indices)  # holder
    dev_size = int(len(test_indices)/2)  # splits up remaining data into two, for 90/5/5, 80/10/10, etc.
    test_size = hold_size - dev_size
    test2_indices = random.sample(test_indices, test_size)
    dev_indices = list(get_dif(test_indices, test2_indices))  # gets the dev indices
    # print(int(len(test_indices)/2))
    return train_indices, test2_indices, dev_indices


def write_to_file(output_file, positive_list, negative_list, positive_idx, negative_idx):
    with open(output_file, "w", newline='', encoding="utf-8") as my_csv:  # create training data file
        my_writer = csv.writer(my_csv)

        for i in positive_idx:  # for all positive indices
            my_writer.writerow(positive_list[i][1:3])  # [1:3] gets the second and third columns from the csv data
        for j in negative_idx:  # for all negative indices
            my_writer.writerow(negative_list[j][1:3])
    my_csv.close()


def main():
    # ---------------- PARAMETERS ----------------
    input_train_data = "hahackathon_train.csv"
    output_train_data = "hahackathon_prepo1_train.csv"
    output_test_data = "hahackathon_prepo1_test.csv"
    output_dev_data = "hahackathon_prepo1_dev.csv"
    # create_dev = False
    # dev_size = 100
    random.seed(573)
    training_ratio = 0.8
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
    train_pos_idx, test_pos_idx, dev_pos_idx = get_indices(positive_list, train_size, rows)  # get humorous indices
    train_neg_idx, test_neg_idx, dev_neg_idx = get_indices(negative_list, train_size, rows)  # get non-humorous indices

    train_pos_idx.sort()  # sort for easier viewing, not necessary
    test_pos_idx.sort()
    dev_pos_idx.sort()

    train_neg_idx.sort()  # sort for easier viewing, not necessary
    test_neg_idx.sort()
    dev_neg_idx.sort()

    write_to_file(output_train_data, positive_list, negative_list, train_pos_idx, train_neg_idx)
    write_to_file(output_test_data, positive_list, negative_list, test_pos_idx, test_neg_idx)
    write_to_file(output_dev_data, positive_list, negative_list, dev_pos_idx, dev_neg_idx)



main()
