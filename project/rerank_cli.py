import argparse
import train_cli
import data


def main_cli():
    parser = argparse.ArgumentParser('CEDR model re-ranking')
    parser.add_argument('--model', choices=train_cli.MODEL_MAP.keys(), default='vanilla_bert')
    parser.add_argument('--datafiles', type=argparse.FileType('rt'), nargs='+')
    parser.add_argument('--run', type=argparse.FileType('rt'))
    parser.add_argument('--model_weights', type=argparse.FileType('rb'))
    parser.add_argument('--out_path', type=argparse.FileType('wt'))
    args = parser.parse_args()
    model = train_cli.MODEL_MAP[args.model]().cuda()
    dataset = data.read_datafiles(args.datafiles)
    run = data.read_run_dict(args.run)
    if args.model_weights is not None:
        model.load(args.model_weights.name)
    rerank_run = train_cli.run_model(model, dataset, run, desc='rerank')
    train_cli.write_run(rerank_run, args.out_path.name)

if __name__ == '__main__':
    main_cli()
