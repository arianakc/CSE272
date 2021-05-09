# Code Written by Changmao Li for CSE272 hw1
# Date: April 16

OHSU_SEQID = ".I"
OHSU_DOCID = ".U"
OHSU_SUBJECT = ".S"
OHSU_MESH = ".M"
OHSU_TITLE = ".T"
OHSU_TYPE = ".P"
OHSU_BODY = ".W"
OHSU_AUTHORS = ".A"
doc_keys = [OHSU_SEQID, OHSU_DOCID, OHSU_SUBJECT, OHSU_MESH, OHSU_TITLE, OHSU_TYPE, OHSU_BODY, OHSU_AUTHORS]


def read_ohsumed(filepath):
    documents = []
    document = {}
    current_key = None
    with open(filepath, "r") as datafile:
        ohsumed_data = datafile.readlines()
        for line in ohsumed_data:
            if line.startswith(OHSU_SEQID):
                if len(document) != 0:
                    documents.append(document)
                    document = {}
                document[OHSU_SEQID] = line.split(" ")[1].strip()
            else:
                if line[0] == '.' and line[1].isupper():
                    text = line.strip()
                    if text in doc_keys:
                        current_key = text
                    else:
                        print("Invalid field name: " + text + ", skipping ...")
                        current_key = None
                    continue
                else:
                    document[current_key] = line.strip()
    if len(document) != 0:
        documents.append(document)
    return documents

# trec format
# qid 0 docno relevance
# Where:
#   qid is the query number
#   0 is the literal 0
#   docno is the id of a document in your collection
#   relevance is how relevant is docno for qid

def read_qrels_to_trec(filepath):
    qrels = []
    with open(filepath) as qrelsfile:
        qrels_data = qrelsfile.readlines()
        for line in qrels_data:
            values = line.strip().split("\t")
            if len(values) < 2:
                print("-invalid line, skipping: " + line)
                continue
            qrel = {}
            qrel["qid"] = values[0]
            qrel["literal"] = "0"
            qrel["docno"] = values[1]
            if len(values) > 2:
                qrel["relevance"] = 2
            else:
                qrel["relevance"] = 1
            qrels.append(qrel)
    return qrels


def output_qrels_to_file(qrels, filepath):
    with open(filepath, "w") as qrels_file:
        for qrel in qrels:
            qrels_file.write(" ".join([qrel["qid"], qrel["literal"], qrel["docno"], str(qrel["relevance"])]))
            qrels_file.write("\n")


def output_results_to_file(results, filepath):
    with open(filepath, "w") as results_file:
        for result in results:
            results_file.write(" ".join([result["QueryID"], result["Q0"], result["DocID"], result["Rank"], result["Score"], result["RunID"]]))
            results_file.write("\n")


def read_queries(filepath):
    queries = []
    query = {}
    description = False
    with open(filepath, "r") as query_file:
        query_data = query_file.readlines()
        for line in query_data:
            line_s = line.rstrip()
            if line_s == "" or line_s == "</top>":
                continue
            elif line_s == "<top>":
                if len(query) != 0:
                    queries.append(query)
                    query = {}
                    continue
            elif line_s.startswith("<num> Number: "):
                query["Number"] = line_s[14:]
            elif line_s.startswith("<title> "):
                query["title"] = line_s[8:].replace("/", " ")
            elif line_s == "<desc> Description:":
                description = True
                continue
            elif description:
                query["description"] = line_s.replace("/", " ")
                description = False
            else:
                print("Unrecognized line, skipping: '" + line + "'")
                continue
    if len(query) != 0:
        queries.append(query)
    return queries





if __name__ == "__main__":
    documents = read_ohsumed("../data/ohsumed.88-91")
    # print(len(documents))
    # print(documents[-1])
    # print("test!")
    qrels = read_qrels_to_trec("../data/qrels.ohsu.88-91")
    print(len(qrels))
    output_qrels_to_file(qrels, "../data/qrels.ohsu.88-91.trec")
    # queries = read_queries("../data/query.ohsu.1-63")
    # print(len(queries))