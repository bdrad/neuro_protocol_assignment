from random import shuffle
from csv import DictReader, DictWriter
import argparse
import numpy as np

from models import get_log_reg_model

CONCAT_ORDER = True

def join_img_order(report, img_order):
    if CONCAT_ORDER:
        return report + " IMGORD_" + img_order.replace(" ", "_")
    else:
        return report

# Returns list of tuples of form (Report Text, Image Order, Protocol Category, Protocol Specific)
def read_csv(in_path):
    csv_reader = DictReader(open(in_path, 'r'))
    valid_entries = filter(lambda entry: entry['PROTOCOL CATEGORY'] != '' and entry['PROTOCOL CATEGORY'] != 'NA', csv_reader)
    return [(join_img_order(e['Final Text'], e['Image Order']), e['Image Order'], e['PROTOCOL CATEGORY'], e['PROTCOL SPECIFIC']) for e in valid_entries]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str)
    parser.add_argument('output_path', type=str)
    args = parser.parse_args()

    
    # Read data and build specific -> general lookup
    data = read_csv(args.input_path)
    spec_to_gen = {}
    for _, _, gen, spec in data:
        if spec not in spec_to_gen.keys():
            spec_to_gen[spec] = gen

    train_ratio = 0.8
    split_point = int(len(data) * train_ratio)
    shuffle(data)
    train_data, test_data = data[:split_point], data[split_point:]
    train_reports, _, _, train_labels = zip(*train_data)
    test_reports, _, _, test_labels = zip(*test_data)

    # Try predicting general label
    logreg_model, _ = get_log_reg_model(train_reports, train_labels)
    print("Score on validation set")
    print(logreg_model.score(test_reports, test_labels))

    # Train a model on all data
    reports, _, _, labels = zip(*data)
    logreg_model, cross_val_score = get_log_reg_model(reports, labels)
    print("Cross Validation Scores")
    print(cross_val_score)
    print(np.mean(cross_val_score))


    # Replace entries
    csv_reader = DictReader(open(args.input_path, 'r'))
    entries = []
    dict_keys = csv_reader.fieldnames + ['Manually Annotated']
    for entry in csv_reader:
        if entry['PROTOCOL CATEGORY'] != '':
            entry['Manually Annotated'] = 1
        else:
            if CONCAT_ORDER:
                pred = logreg_model.predict([entry['Final Text'] + " " + entry['Image Order']])[0]
            else:
                pred = logreg_model.predict([entry['Final Text']])[0]
            entry['PROTCOL SPECIFIC'] = pred
            entry['PROTOCOL CATEGORY'] = spec_to_gen[pred]
            entry['Manually Annotated'] = 0
        entries.append(entry)
    
    writer = DictWriter(open(args.output_path, 'w'), fieldnames=dict_keys)
    writer.writeheader()
    writer.writerows(entries)

if __name__ == "__main__":
    main()