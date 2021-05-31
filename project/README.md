

#### Prepare Environment

```bash
pip3 install -r requirements.txt
```

#### Data preparation
Download and unzip data from http://www.cs.jhu.edu/~shuosun/clirmatrix/ to data director and run:

```bash
python3 preprocess.py
```

#### Training

```bash
./train_<model>_en_<doc_lang>.sh
```

#### Prediction Ranking on Test

```bash
./rerank_<model>_en_<doc_lang>.sh
```

#### Evaluation

```bash
./trec_eval -m ndcg.cut.10 data/en.<target_lang>.qrels models/<model>_<doc_lang>/en.<doc_lang>.test.run
```
