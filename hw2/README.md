### Environment
- Python3
- [Surprise](https://github.com/NicolasHug/Surprise)
- required packages: tqdm, joblib>=0.11, numpy>=1.11.2, scipy>=1.0.0,six>=1.10.0
### Data processor

```bash
usage: data_processor.py [-h] [--data_file_path DATA_FILE_PATH] [--train_file_path TRAIN_FILE_PATH] [--test_file_path TEST_FILE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --data_file_path DATA_FILE_PATH
                        data file path
  --train_file_path TRAIN_FILE_PATH
                        training file path
  --test_file_path TEST_FILE_PATH
                        testing file path

```
#### Example Usage
```bash
python3 data_processor.py --data_file_path data/reviews_Movies_and_TV_5.json --train_file_path data/train.csv --test_file_path data/test.csv
```

### Recommender

```bash
usage: recommender.py [-h] [--train_file_path TRAIN_FILE_PATH] [--test_file_path TEST_FILE_PATH] [--approach APPROACH] [--output_ranking_file OUTPUT_RANKING_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --train_file_path TRAIN_FILE_PATH
                        training file path
  --test_file_path TEST_FILE_PATH
                        testing file path
  --approach APPROACH   BaselineOnly | SVD | SlopeOne | NMF | CoClustering
  --output_ranking_file OUTPUT_RANKING_FILE
                        output ranking for test
```
#### Example Usage
```bash
python3 recommender.py --train_file_path data/train.csv --test_file_path data/test.csv --approach BaselineOnly --output_ranking_file ranking
```





