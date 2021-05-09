# Code written by Changmao Li for CSE272 hw1
# Date Created: April 17

from src.data_processor import read_queries, read_ohsumed, read_qrels_to_trec, output_results_to_file
import sys, lucene
from java.nio.file import Paths
from org.apache.lucene.document import Document, Field, StoredField, StringField, TextField, NumericDocValuesField, FieldType
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, Term, IndexReader, DirectoryReader, IndexOptions, PostingsEnum, TermsEnum
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.core import WhitespaceAnalyzer
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.store import Directory, FSDirectory
from org.apache.lucene.util import Version
from org.apache.lucene.search.similarities import BooleanSimilarity, ClassicSimilarity, SimilarityBase, BM25Similarity, \
    LMDirichletSimilarity, LMJelinekMercerSimilarity, MultiSimilarity
from org.apache.pylucene.search.similarities import PythonClassicSimilarity
from org.apache.lucene.search import BooleanClause, BooleanQuery, TermQuery, IndexSearcher, Explanation
from collections import Counter
from math import log10, log2
from org.apache.lucene.util import BytesRefIterator
from tqdm import tqdm
import spacy_sentence_bert
import numpy as np
from scipy import spatial
import math
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import gensim.downloader
tokenizer = nltk.RegexpTokenizer(r"\w+")
def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec

def stop_words():
    nltk.download('stopwords')
    return list(set(stopwords.words('english')))


class IndexBuilder(object):
    def __init__(self, index_path, update=False):
        dir = FSDirectory.open(Paths.get(index_path))
        analyzer = StandardAnalyzer()
        iwc = IndexWriterConfig(analyzer)
        if update:
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND)
        else:
            iwc.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        self.writer = IndexWriter(dir, iwc)

    def index_docs(self, input_documents):
        for document in tqdm(input_documents, total=len(input_documents)):
            doc = Document()
            doc.add(StringField(".I", document[".I"].lower(), Field.Store.YES))
            doc.add(StringField(".U", document[".U"].lower(), Field.Store.YES))
            type = FieldType()
            type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS)
            type.setStored(True)
            type.setStoreTermVectors(True)
            type.setTokenized(True)
            if ".W" in document and ".M" in document:
                doc.add(Field("text", " ".join(tokenizer.tokenize(document[".M"].lower() + " " + document[".T"].lower() + document[".W"].lower())), type))
            elif ".M" in document and ".W" not in document:
                doc.add(Field("text", " ".join(tokenizer.tokenize(document[".M"].lower() + " " + document[".T"].lower())), type))
            elif ".M" not in document and ".W" in document:
                doc.add(Field("text",   " ".join(tokenizer.tokenize(document[".T"].lower() + document[".W"].lower())), type))
            elif ".M" not in document and ".W" not in document:
                doc.add(Field("text", " ".join(tokenizer.tokenize(document[".T"].lower())), type))
            if self.writer.getConfig().getOpenMode() == IndexWriterConfig.OpenMode.CREATE:
                self.writer.addDocument(doc)
            else:
                self.writer.updateDocument(Term(".U", document[".U"]), doc)
        self.writer.close()



if __name__ == "__main__":
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    # documents = read_ohsumed("../data/ohsumed.88-91")
    # index_builder = IndexBuilder("../lucene-index")
    # index_builder.index_docs(documents)
    # feedback_documents = read_ohsumed("../data/ohsumed.87")
    # index_builder = IndexBuilder("../feedback-lucene-index1")
    # index_builder.index_docs(feedback_documents)
    # search_builder = SearchBuilder("../lucene-index", "text", similarity="BM25",
    #                                use_relevance_feedback=True, feedback_index_path="../feedback-lucene-index1")
    # queries = read_queries("../data/query.ohsu.1-63")
    # trec_results = search_builder.get_results_from_queries(queries, use_multipass_pseudo_relevance_feedback=False)
    # output_results_to_file(trec_results, "../data/tfidfresults.trec")
    # print("Loading glove vectors!")
    # glove_vectors = gensim.downloader.load('glove-wiki-gigaword-300')
    # print("Finished Loading glove vectors!")
    # # fasttext_vectors = gensim.downloader.load("fasttext-wiki-news-subwords-300")
    # from sentence_transformers import SentenceTransformer, util
    #
    # distilroberta_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    # feedback_qrels = read_qrels_to_trec("../data/qrels.ohsu.batch.87")
    # trec_results = search_builder.get_results_from_queries_with_relevance_feedback(queries, feedback_qrels)
    # output_results_to_file(trec_results, "../data/tfidfwithrfresults.trec")
    # doc_vectors = {}
    # print("build doc vectors!")
    # index2word_set = set(glove_vectors.wv.index2word)
    # for i in tqdm(range(search_builder.reader.maxDoc()), total=search_builder.reader.maxDoc()):
    #     doc = search_builder.reader.document(i)
    #     doc_vectors[doc.get(".U")] = avg_feature_vector(doc.get("text"), model=glove_vectors, num_features=300,
    #                                                     index2word_set=index2word_set)
    # trec_results = search_builder.get_results_from_queries_with_pretrained_embedding_similariy(queries, doc_vectors)
    # output_results_to_file(trec_results, "../data/tfidfresults.trec")

    # doc_vectors = {}
    # print("build doc vectors!")
    # for i in tqdm(range(search_builder.reader.maxDoc()), total=search_builder.reader.maxDoc()):
    #     doc = search_builder.reader.document(i)
    #     doc_vectors[doc.get(".U")] = distilroberta_model.encode(doc.get("text"), convert_to_tensor=True)
    # trec_results = search_builder.get_results_from_queries_with_transformers(queries, doc_vectors)
    # output_results_to_file(trec_results, "../data/tfidfresults.trec")