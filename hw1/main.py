# Code written by Changmao Li For CSE 272 hw1
import lucene
import os
from src.index_builder import IndexBuilder
from src.data_processor import read_ohsumed, read_queries, output_results_to_file, read_qrels_to_trec
from src.index_searcher import SearchBuilder
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="search_index", help="build_index | search_index")
    parser.add_argument("--approach", default="BM25", help="boolean | tf | tfidf")
    parser.add_argument("--documents_file", default="data/ohsumed.88-91")
    parser.add_argument("--query_file", default="data/query.ohsu.1-63")
    parser.add_argument("--doc_index_path", default="lucene-index")
    parser.add_argument("--output_file", default="results.trec")
    parser.add_argument("--use_fb", action="store_true",
                        help="Use Relevance Feedback")
    parser.add_argument("--feed_back_doc_file", default="data/ohsumed.87")
    parser.add_argument("--feed_back_doc_index_path", default="feedback-lucene-index1")
    parser.add_argument("--feed_back_qrels_path", default="data/qrels.ohsu.batch.87")
    parser.add_argument("--use_pfb", action="store_true", help="Use Pseudo Relevance Feedback")

    args = parser.parse_args()
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    if args.task == "build_index":
        print("Build Index....")
        documents = read_ohsumed(args.documents_file)
        index_builder = IndexBuilder(args.doc_index_path)
        index_builder.index_docs(documents)
        print("Build Feedback Index....")
        feedback_documents = read_ohsumed(args.feed_back_doc_file)
        index_builder = IndexBuilder(args.feed_back_doc_index_path)
        index_builder.index_docs(feedback_documents)
    if args.task == "search_index":
        start_time = time.time()
        print("Build Searcher...")
        search_builder = SearchBuilder(args.doc_index_path, "text", similarity=args.approach,
                                       use_relevance_feedback=args.use_fb,
                                       feedback_index_path=args.feed_back_doc_index_path)
        queries = read_queries(args.query_file)
        if args.use_fb:
            feedback_qrels = read_qrels_to_trec(args.feed_back_qrels_path)
            trec_results = search_builder.get_results_from_queries_with_relevance_feedback(queries, feedback_qrels)
        else:
            trec_results = search_builder.get_results_from_queries(queries, use_pseudo_relevance_feedback=args.use_pfb)
        print("Output Results...")
        output_results_to_file(trec_results, args.output_file)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"The search cost:{total_time}s")