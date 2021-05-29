import argparse
from tqdm import tqdm
import json


def read_jl(jl_path):
    examples = []
    with open(jl_path, "r") as in_file:
        lines = in_file.readlines()
        for line in tqdm(lines, desc=f"read {jl_path}", total=len(lines)):
            examples.append(json.loads(line.strip("\n")))
    return examples


def generate_queries(train_path, test_path, dev_path, out_path):
    examples = read_jl(train_path) + read_jl(dev_path) + read_jl(test_path)
    with open(out_path, "w") as out_file:
        for example in tqdm(examples, desc=f"generating {out_path}", total=len(examples)):
            out_file.write("query" + "\t" + example["src_id"] + "\t" + example["src_query"] + "\n")


def generate_docs(in_path, out_path):
    with open(out_path, "w") as out_file:
        with open(in_path, "r") as in_file:
            in_lines = in_file.readlines()
            for in_line in tqdm(in_lines, desc=f"generating doc file for {in_path}", total=len(in_lines)):
                out_file.write("doc" + "\t" + in_line)


def generate_trec_qrels(train_path, test_path, dev_path, out_path):
    examples = read_jl(train_path) + read_jl(dev_path) + read_jl(test_path)
    with open(out_path, "w") as out_file:
        for example in tqdm(examples, desc=f"generating {out_path}", total=len(examples)):
            query_id = example["src_id"]
            documents = example["tgt_results"]
            for document in documents:
                document_id = document[0]
                score = document[1]
                out_file.write(query_id+" 0 " + document_id+" "+str(score)+"\n")


def generate_trec_run(in_path, out_path):
    examples = read_jl(in_path)
    with open(out_path, "w") as out_file:
        for example in tqdm(examples, desc=f"generating {out_path}", total=len(examples)):
            query_id = example["src_id"]
            documents = example["tgt_results"]
            for i, document in enumerate(documents):
                document_id = document[0]
                score = document[1]
                if score == 0:
                    break
                out_file.write(query_id + " Q0 " + document_id + " " + str(i + 1) + " " + str(score) + " " + out_path + "\n")



def generate_pairs(in_path, out_path):
    examples = read_jl(in_path)
    with open(out_path, "w") as out_file:
        for example in tqdm(examples, desc=f"generating {out_path}", total=len(examples)):
            query_id = example["src_id"]
            documents = example["tgt_results"]
            for document in documents:
                document_id = document[0]
                out_file.write(query_id+"\t"+document_id+"\n")


if __name__ == '__main__':
    ar_doc_in_path = "data/ar.tsv"
    ar_doc_out_path = "data/ar.documents.tsv"
    zh_doc_in_path = "data/zh.tsv"
    zh_doc_out_path = "data/zh.documents.tsv"
    # generate_docs(ar_doc_in_path, ar_doc_out_path)
    # generate_docs(zh_doc_in_path, zh_doc_out_path)
    en_ar_train = "data/en.ar.train.jl"
    en_ar_dev = "data/en.ar.dev.jl"
    en_ar_test = "data/en.ar.test1.jl"
    en_zh_train = "data/en.zh.train.jl"
    en_zh_dev = "data/en.zh.dev.jl"
    en_zh_test = "data/en.zh.test1.jl"
    en_ar_queries = "data/en.ar.queries.tsv"
    en_zh_queries = "data/en.zh.queries.tsv"
    # generate_queries(en_ar_train, en_ar_dev, en_ar_test, en_ar_queries)
    # generate_queries(en_zh_train, en_zh_dev, en_zh_test, en_zh_queries)
    en_ar_train_pairs = "data/en.ar.train.pairs"
    en_zh_train_pairs = "data/en.zh.train.pairs"
    # generate_pairs(en_ar_train, en_ar_train_pairs)
    # generate_pairs(en_zh_train, en_zh_train_pairs)
    en_ar_qrels = "data/en.ar.qrels"
    en_zh_qrels = "data/en.zh.qrels"
    # generate_trec_qrels(en_ar_train, en_ar_dev, en_ar_test, en_ar_qrels)
    generate_trec_qrels(en_zh_train, en_zh_dev, en_zh_test, en_zh_qrels)
    en_ar_dev_run = "data/en.ar.dev.run"
    en_zh_dev_run = "data/en.zh.dev.run"
    en_ar_test_run = "data/en.ar.test.run"
    en_zh_test_run = "data/en.zh.test.run"
    # generate_trec_run(en_ar_dev, en_ar_dev_run)
    # generate_trec_run(en_ar_test, en_ar_test_run)
    # generate_trec_run(en_zh_dev, en_zh_dev_run)
    # generate_trec_run(en_zh_test, en_zh_test_run)
    print("Finished!")

