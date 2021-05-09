from java.nio.file import Paths
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, Term, IndexReader, DirectoryReader, IndexOptions, PostingsEnum, TermsEnum
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import Directory, FSDirectory
from org.apache.lucene.search.similarities import BooleanSimilarity, ClassicSimilarity, SimilarityBase, BM25Similarity, \
    LMDirichletSimilarity, LMJelinekMercerSimilarity, MultiSimilarity
from org.apache.pylucene.search.similarities import PythonClassicSimilarity
from org.apache.lucene.search import BooleanClause, BooleanQuery, TermQuery, IndexSearcher, Explanation
from org.apache.lucene.util import BytesRefIterator
import math
import nltk
from nltk.corpus import stopwords
tokenizer = nltk.RegexpTokenizer(r"\w+")

def stop_words():
    nltk.download('stopwords')
    return list(set(stopwords.words('english')))

class SearchBuilder(object):
    def __init__(self, index_path, field, similarity="boolean", use_relevance_feedback=False, feedback_index_path=None):
        self.reader = DirectoryReader.open(FSDirectory.open(Paths.get(index_path)))
        self.searcher = IndexSearcher(self.reader)
        if use_relevance_feedback and feedback_index_path is not None:
            self.feedback_reader = DirectoryReader.open(FSDirectory.open(Paths.get(feedback_index_path)))
            self.feedback_searcher = IndexSearcher(self.feedback_reader)
        self.similarity = similarity
        self.stopwords = stop_words()
        if similarity == "boolean":
            self.searcher.setSimilarity(BooleanSimilarity())
        elif similarity == "tf":
            self.searcher.setSimilarity(TFSimilarity())
        elif similarity == "tfidf":
            self.searcher.setSimilarity(ClassicSimilarity())
        elif similarity == "BM25":
            self.searcher.setSimilarity(BM25Similarity(1.2, 0.2))
        else:
            print("Unknown similarity, so we use BM25(1.2, 0.2) as default")
            self.searcher.setSimilarity(BM25Similarity(1.2, 0.2))
        analyzer = StandardAnalyzer()
        print(self.searcher.getSimilarity())
        self.parser = QueryParser(field, analyzer)

    def remove_stopwords(self, query_text):
        new_query_tokens = []
        query_tokens = query_text.split()
        for query_token in query_tokens:
            if query_token not in self.stopwords:
                new_query_tokens.append(query_token)
        return " ".join(new_query_tokens)

    def search_query(self, query, num_returns=50, use_multipass_pseudo_relevance_feedback=False, doc_counts=None, add_nums=None):

        query_text = query["description"]
        print(query_text.lower())
        query_text = " ".join(tokenizer.tokenize(query_text))
        query_text = self.remove_stopwords(query_text.lower())
        print(query_text)
        query_search = self.parser.parse(query_text)
        if use_multipass_pseudo_relevance_feedback:
            if doc_counts is None:
                doc_counts = [5, 9]
            if add_nums is None:
                add_nums = [2, 13]
            assert len(doc_counts) == len(add_nums), "The number of pass is inconsistent!"
            for doc_count, add_num in zip(doc_counts, add_nums):
                final_list = []
                initial_hits = self.searcher.search(query_search, doc_count).scoreDocs
                term_tf_idf = {}
                for initial_hit in initial_hits:
                    termVector = self.reader.getTermVector(initial_hit.doc, "text")
                    terms_enum = termVector.iterator()
                    termsref = BytesRefIterator.cast_(terms_enum)
                    N_terms = 0
                    term_idf = {}
                    term_freq = {}
                    term_list = []
                    while (termsref.next()):
                        termval = TermsEnum.cast_(termsref)
                        termText = termval.term().utf8ToString()
                        if termText in self.stopwords:
                            continue
                        tc = termval.totalTermFreq()
                        if termText in term_freq:
                            term_freq[termText] += tc
                        else:
                            term_freq[termText] = tc
                        if termText in term_idf:
                            term_idf[termText] += 1
                        else:
                            term_idf[termText] = 1
                        if termText not in term_list:
                            term_list.append(termText)
                        N_terms = N_terms + 1

                    for term in term_list:
                        if term in term_tf_idf:
                            term_tf_idf[term] += term_freq[term] / N_terms * (
                                    1 + math.log(doc_count / (term_idf[term] + 1)))
                        else:
                            term_tf_idf[term] = term_freq[term] / N_terms * (
                                        1 + math.log(doc_count / (term_idf[term] + 1)))
                sorted_term_tf_idf = sorted(term_tf_idf.items(), key=lambda x: x[1], reverse=True)
                for each in sorted_term_tf_idf:
                    if each[0] not in self.stopwords:
                        final_list.append(each[0])
                print("added query tokens:", final_list[:add_num])
                query_text = query_text + " " + " ".join(final_list[:add_num])
                query_search = self.parser.parse(query_text)
        results = self.searcher.search(query_search, num_returns)
        hits = results.scoreDocs
        trec_results = []
        for rank, hit in enumerate(hits):
            doc = self.searcher.doc(hit.doc)
            trec_result = {"QueryID": query["Number"],
                           "Q0": "Q0",
                           "DocID": doc.get(".U"),
                           "Rank": str(rank + 1),
                           "Score": str(hit.score),
                           "RunID": self.similarity + "-mpprf-"+str(len(doc_counts))+"passes" if use_multipass_pseudo_relevance_feedback else self.similarity}
            trec_results.append(trec_result)
        return trec_results

    def search_query_with_relevance_feedback(self, query, feedback_qrels, num_returns=50, add_num=1):
        query_text = query["description"]
        print(query_text)
        query_text = " ".join(tokenizer.tokenize(query_text))
        query_text = self.remove_stopwords(query_text.lower())
        print(query_text)
        query_number = query["Number"]
        qrel_doc_ids = [qrel["docno"] for qrel in feedback_qrels if qrel["qid"] == query_number]
        final_list = []
        term_tf_idf = {}
        doc_count = len(qrel_doc_ids)
        for qrel_doc_id in qrel_doc_ids:
            initial_hit = self.feedback_searcher.search(TermQuery(Term(".U", qrel_doc_id)), 1).scoreDocs
            if len(initial_hit) == 0:
                continue
            assert len(initial_hit) == 1
            termVector = self.reader.getTermVector(initial_hit[0].doc, "text")
            terms_enum = termVector.iterator()
            termsref = BytesRefIterator.cast_(terms_enum)
            N_terms = 0
            term_idf = {}
            term_freq = {}
            term_list = []
            while (termsref.next()):
                termval = TermsEnum.cast_(termsref)
                termText = termval.term().utf8ToString()
                if termText in self.stopwords:
                    continue
                tc = termval.totalTermFreq()
                if termText in term_freq:
                    term_freq[termText] += tc
                else:
                    term_freq[termText] = tc
                if termText in term_idf:
                    term_idf[termText] += 1
                else:
                    term_idf[termText] = 1
                if termText not in term_list:
                    term_list.append(termText)
                N_terms = N_terms + 1

            for term in term_list:
                if term in term_tf_idf:
                    term_tf_idf[term] += term_freq[term] / N_terms * (1+math.log(doc_count / (term_idf[term] + 1)))
                else:
                    term_tf_idf[term] = term_freq[term] / N_terms * (1 + math.log(doc_count / (term_idf[term] + 1)))

        sorted_tf_idf = sorted(term_tf_idf.items(), key=lambda x: x[1], reverse=True)
        for each in sorted_tf_idf:
            if each[0] not in self.stopwords and not str(each[0]).isnumeric() and each[0] not in query_text.split(" "):
                final_list.append(each[0])
        print(final_list[:add_num])
        query_text = query_text + " " + " ".join(final_list[:add_num])
        query_text = " ".join(query_text.split(" "))
        print(query_text)
        query_search = self.parser.parse(query_text)
        results = self.searcher.search(query_search, num_returns)
        hits = results.scoreDocs
        trec_results = []
        for rank, hit in enumerate(hits):
            doc = self.searcher.doc(hit.doc)
            trec_result = {"QueryID": query["Number"],
                           "Q0": "Q0",
                           "DocID": doc.get(".U"),
                           "Rank": str(rank + 1),
                           "Score": str(hit.score),
                           "RunID": self.similarity}
            trec_results.append(trec_result)
        return trec_results

    # def search_query_with_glove(self,  query, doc_vectors, num_returns=50, index2word_set=None):
    #     query_text = query["description"]
    #     query_text = " ".join(word_tokenize(query_text))
    #     query_text = self.remove_stopwords(query_text)
    #     query_vec = avg_feature_vector(query_text, model=glove_vectors, num_features=300, index2word_set=index2word_set)
    #     doc_similarity = {}
    #     for doc_id in tqdm(doc_vectors, desc="compute doc similarity:", total=len(doc_vectors.items())):
    #         doc_similarity[doc_id] = 1 - spatial.distance.cosine(query_vec, doc_vectors[doc_id])
    #     doc_similarity = sorted(doc_similarity.items(), key=lambda x: x[1], reverse=True)[:num_returns]
    #     trec_results = []
    #     for i, doc_id in tqdm(enumerate(doc_similarity), desc="output results:", total=len(doc_similarity)):
    #         trec_result = {"QueryID": query["Number"],
    #                        "Q0": "Q0",
    #                        "DocID": doc_id[0],
    #                        "Rank": str(i + 1),
    #                        "Score": str(doc_id[1]),
    #                        "RunID": self.similarity+"+embedding"}
    #         trec_results.append(trec_result)
    #     return trec_results
    #
    # def search_query_with_transformers(self,  query, doc_vectors, num_returns=50):
    #     query_text = query["description"]
    #     query_text = " ".join(word_tokenize(query_text))
    #     query_text = self.remove_stopwords(query_text)
    #     query_vec = distilroberta_model.encode(query_text, convert_to_tensor=True)
    #     doc_similarity = {}
    #     for doc_id in tqdm(doc_vectors, desc="compute doc similarity:", total=len(doc_vectors.items())):
    #         doc_similarity[doc_id] = util.pytorch_cos_sim(query_vec, doc_vectors[doc_id])
    #     doc_similarity = sorted(doc_similarity.items(), key=lambda x: x[1], reverse=True)[:num_returns]
    #     trec_results = []
    #     for i, doc_id in tqdm(enumerate(doc_similarity), desc="output results:", total=len(doc_similarity)):
    #         trec_result = {"QueryID": query["Number"],
    #                        "Q0": "Q0",
    #                        "DocID": doc_id[0],
    #                        "Rank": str(i + 1),
    #                        "Score": str(doc_id[1]),
    #                        "RunID": self.similarity+"+embedding"}
    #         trec_results.append(trec_result)
    #     return trec_results

    def get_results_from_queries(self, queries, num_returns=50, use_pseudo_relevance_feedback=False):
        trec_results = []
        for query in queries:
            search_results = self.search_query(query, num_returns, use_pseudo_relevance_feedback)
            trec_results = trec_results + search_results
        return trec_results
    #
    # def get_results_from_queries_with_pretrained_embedding_similariy(self, queries, doc_vectors, num_returns=50):
    #     trec_results = []
    #     for query in tqdm(queries, desc="queries", total=len(queries)):
    #         search_results = self.search_query_with_glove(query, doc_vectors, num_returns)
    #         trec_results = trec_results + search_results
    #     return trec_results
    #
    # def get_results_from_queries_with_transformers(self, queries, doc_vectors, num_returns=50):
    #     trec_results = []
    #     for query in tqdm(queries, desc="queries", total=len(queries)):
    #         search_results = self.search_query_with_transformers(query, doc_vectors, num_returns)
    #         trec_results = trec_results + search_results
    #     return trec_results

    def get_results_from_queries_with_relevance_feedback(self, queries, feedback_qrels, num_returns=50):
        trec_results = []
        for query in queries:
            search_results = self.search_query_with_relevance_feedback(query, feedback_qrels, num_returns=num_returns)
            trec_results = trec_results + search_results
        return trec_results


class TFSimilarity(PythonClassicSimilarity):

    def lengthNorm(self, numTerms):
        return (float)(1.0 / math.sqrt(numTerms))

    def tf(self, freq):
        return math.sqrt(freq) * 1.0

    def idf(self, docFreq, numDocs):
        return 1.0

