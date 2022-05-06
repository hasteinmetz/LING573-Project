import nltk
import csv
import sys


def write_to_file(output_file, named_entity_list):
    with open(output_file, "w", newline='', encoding="utf-8") as my_csv:  # create training data file
        my_writer = csv.writer(my_csv)
        for named_entity in named_entity_list:
            my_writer.writerow(named_entity)
    my_csv.close()


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    correct_label_column = int(sys.argv[3])
    ne_groupings = []
    # print("here!")
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

            if len(ne_list) > 0:
                temp = []
                for item in ne_list:
                    group = []
                    # print(item)
                    for j in range(len(item)):
                        group.append(item[j][0])  # get the object recognized as NE
                    temp.append(" ".join(group))
                temp.append(int(row[correct_label_column]))  # correct_label, CHANGE DEPENDING ON INPUT FILE!
                ne_groupings.append(temp)

    csv_file.close()
    write_to_file(output_file, ne_groupings)


main()
