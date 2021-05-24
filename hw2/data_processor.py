import json
import random
from tqdm import tqdm
import argparse


def read_json(data_file_path):
    data = []
    print(f"read {data_file_path}...")
    with open(data_file_path, "r") as data_file:
        lines = data_file.readlines()
        for line in tqdm(lines, desc=f"read {data_file_path}", total=len(lines)):
            data.append(json.loads(line))
    return data


def split_data(data):
    user_ids_examples = {}
    train_data = []
    test_data = []
    for example in tqdm(data, desc="generate user ids examples...", total=len(data)):
        if example["reviewerID"] not in user_ids_examples:
            user_ids_examples[example["reviewerID"]] = [example]
        else:
            user_ids_examples[example["reviewerID"]].append(example)
    for user_ids in tqdm(user_ids_examples, desc="generate data split...", total=len(user_ids_examples)):
        examples = user_ids_examples[user_ids]
        random.shuffle(examples)
        train_size = int(len(examples)*0.8)
        train_data.extend(examples[0:train_size])
        test_data.extend(examples[train_size:])
    return train_data, test_data


def convert_json_to_user_item_rating_csv(data, csv_file):
    with open(csv_file, "w") as out_csv:
        for example in tqdm(data, total=len(data), desc=f"generate {csv_file}"):
            user_id = example["reviewerID"]
            item_id = example["asin"]
            rating = example["overall"]
            time = example["unixReviewTime"]
            out_csv.write(user_id+"\t"+item_id+"\t"+str(rating)+"\t"+str(time)+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_path", default="data/reviews_Movies_and_TV_5.json", help="data file path")
    parser.add_argument("--train_file_path", default="data/train.csv", help="training file path")
    parser.add_argument("--test_file_path", default="data/test.csv", help="testing file path")
    args = parser.parse_args()
    dataset = read_json(args.data_file_path)
    train_data, test_data = split_data(dataset)
    convert_json_to_user_item_rating_csv(train_data, args.train_file_path)
    convert_json_to_user_item_rating_csv(test_data, args.test_file_path)