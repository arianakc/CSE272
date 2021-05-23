from surprise import BaselineOnly, SVD, SlopeOne, NMF, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from metrics import precision_recall_at_k, get_conversion_rate, get_ndcg
from utils import output_ranking
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="data/train.csv", help="training file path")
    parser.add_argument("--test_file_path", default="data/test.csv", help="testing file path")
    parser.add_argument("--approach", default="SVD", help="BaselineOnly | SVD | SlopeOne | NMF | CoClustering")
    parser.add_argument("--output_ranking_file", default="ranking", help="output ranking for test")
    options = {"BaselineOnly": BaselineOnly, "SVD": SVD, "SlopeOne": SlopeOne, "NMF": NMF, "CoClustering": CoClustering}
    args = parser.parse_args()
    train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    algo = options[args.approach]()
    train_data = Dataset.load_from_file(train_file_path, reader=reader)
    test_data = Dataset.load_from_file(test_file_path, reader=reader)
    train_set = train_data.build_full_trainset()
    test_set = test_data.build_full_trainset().build_testset()
    print("training....")
    algo.fit(train_set)
    print("testing...")
    predictions = algo.test(test_set)
    accuracy.mae(predictions, verbose=True)
    accuracy.rmse(predictions, verbose=True)
    output_ranking(predictions, args.output_ranking_file + "_" + args.approach + ".out")
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=2.5)
    print("precision:", sum(prec for prec in precisions.values()) / len(precisions))
    print("recall:", sum(rec for rec in recalls.values()) / len(recalls))
    print("conversion_rate:", get_conversion_rate(predictions, k=10))
    print("ndcg:", get_ndcg(predictions, k_highest_scores=10))


if __name__ == '__main__':
    main()
