from surprise import BaselineOnly, SVD, SlopeOne, NMF, CoClustering
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from metrics import precision_recall_at_k, get_conversion_rate, get_ndcg, f_measure
from utils import output_ranking
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", default="data/train.csv", help="training file path")
    parser.add_argument("--test_file_path", default="data/test.csv", help="testing file path")
    parser.add_argument("--approach", default="SVD", help="Baseline | SVD | SlopeOne | NMF | CoClustering")
    parser.add_argument("--output_ranking_file", default="ranking", help="output ranking for test")
    bsl_options = {'method': 'sgd', 'n_epochs': 20, 'reg_u': 100, 'reg_i': 50}
    options = {"Baseline": BaselineOnly(bsl_options, verbose=True),
               "SVD": SVD(verbose=True, n_factors=20, n_epochs=3),
               "SlopeOne": SlopeOne(),
               "NMF": NMF(),
               "CoClustering": CoClustering()}
    args = parser.parse_args()
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    algo = options[args.approach]
    train_data = Dataset.load_from_file(args.train_file_path, reader=reader)
    test_data = Dataset.load_from_file(args.test_file_path, reader=reader)
    train_set = train_data.build_full_trainset()
    test_set = test_data.build_full_trainset().build_testset()
    print("training....")
    algo.fit(train_set)
    print("testing...")
    predictions = algo.test(test_set)
    accuracy.mae(predictions, verbose=True)
    accuracy.rmse(predictions, verbose=True)
    ### Extra Credit
    output_ranking(predictions, args.output_ranking_file + "_" + args.approach + ".out")
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=2.5)
    print("Precision:", sum(prec for prec in precisions.values()) / len(precisions))
    print("Recall:", sum(rec for rec in recalls.values()) / len(recalls))
    print("F-measure:", f_measure(precisions, recalls))
    print("conversion_rate:", get_conversion_rate(predictions, k=10))
    print("ndcg:", get_ndcg(predictions, k_highest_scores=10))


if __name__ == '__main__':
    main()
