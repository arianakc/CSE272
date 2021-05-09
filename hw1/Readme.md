"bm25pfb2pass-results.trec" is the best performing results file


### Usage:
#### Set up the environment
- Install the python3
- Install the pylucene based on the instructions from https://lucene.apache.org/pylucene/
- Install other required python package
    - nltk
    - tqdm
    - numpy
    - spacy_sentence_bert
    - scipy
    - gensim
    - spacy_sentence_bert
    
#### Parameters for main
```bash
usage: main.py [-h] [--task TASK] [--approach APPROACH] [--documents_file DOCUMENTS_FILE] [--query_file QUERY_FILE] [--doc_index_path DOC_INDEX_PATH] [--output_file OUTPUT_FILE] [--use_fb]
               [--feed_back_doc_file FEED_BACK_DOC_FILE] [--feed_back_doc_index_path FEED_BACK_DOC_INDEX_PATH] [--feed_back_qrels_path FEED_BACK_QRELS_PATH] [--use_pfb]

optional arguments:
  -h, --help            show this help message and exit
  --task TASK           build_index | search_index(you have to first build index)
  --approach APPROACH   boolean | tf | tfidf | BM25
  --documents_file DOCUMENTS_FILE
                        documents file path
  --query_file QUERY_FILE
                        query file path
  --doc_index_path DOC_INDEX_PATH
                        documents index path
  --output_file OUTPUT_FILE
                        output results file
  --use_fb              Use Relevance Feedback
  --feed_back_doc_file FEED_BACK_DOC_FILE
                        feedback documents file path
  --feed_back_doc_index_path FEED_BACK_DOC_INDEX_PATH
                        feedback index file path
  --feed_back_qrels_path FEED_BACK_QRELS_PATH
                        feedback qrels path
  --use_pfb             Use Multipass Pseudo Relevance Feedback
```

#### The commands to reproduce the best results file:
```bash
python3 main.py --approach BM25 --output_file bm25pfb2pass-results.trec --use_pfb
```
#### The commands to eval the best results file using trec_eval:
```bash
 trec_eval -c -m recall -m recall.50 -m P.50 -m map data/qrels.ohsu.88-91.trec data/bm25pfd1pass-results.trec
```
qrels.ohsu.88-91.trec  is the converted trec format gold answer file using data processor.


