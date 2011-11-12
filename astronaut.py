import string
import sys
import argparse
import textwrap
from gensim.corpora.textcorpus import getstream
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities, interfaces


DONE_WORDS = ["done", "exit", "quit", "q"]
PUNCT_CHARS = "!@#$%^&*`,.:;'\"[](){}"

def make_trans_table(trans_from, trans_to, remove=""):
    trans_tab = dict(zip(map(ord, trans_from), map(ord, trans_to)))
    trans_tab.update((ord(c), None) for c in remove)
    return trans_tab

PUNCT_TRANS_TABLE = make_trans_table('-', " ", remove=PUNCT_CHARS)

def make_tokenize_line_func(trans_table=PUNCT_TRANS_TABLE):
    def tokenize_line(line, trans_table=trans_table):
        line = unicode(line).translate(trans_table)
        line = line.lower().split()
        return line
    return tokenize_line

default_tokenize_line = make_tokenize_line_func()


class OptionalDictCorpus(interfaces.CorpusABC):
    """A corpus that can create its own dictionary or have another dictionary specified,
    which is useful when you want to compare queries against a corpus in an already
    computed space.
    """
    
    def __init__(self, input, dictionary=None, tokenize_line_func=default_tokenize_line, stopwords=stopwords.words('english')):
        super(OptionalDictCorpus, self).__init__()
        self.stoplist = set(stopwords)
        self.input = input
        self.tokenize_line = tokenize_line_func

        if dictionary:
            self.dictionary = dictionary
        else:
            self.dictionary = self.ibuild_dictionary()

        print "%d lines in corpus." % len(self)

    def getstream(self):
        return getstream(self.input)

    def get_texts(self):
        """
        Iterate over the collection, yielding one document at a time. A document
        is a sequence of words (strings) that can be fed into `Dictionary.doc2bow`.

        Override this function to match your input (parse input files, do any
        text preprocessing, lowercasing, tokenizing etc.). There will be no further
        preprocessing of the words coming out of this function.
        """
        for lineno, line in enumerate(getstream(self.input)):
            yield self.tokenize_line(line)

    def ibuild_dictionary(self):
        dictionary = corpora.Dictionary(self.tokenize_line(line) for line in self.get_texts())

        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify() # remove gaps in id sequence after words that were removed        

        print "%d terms in dictionary." % len(dictionary)
        return dictionary
            
    def __iter__(self):
        for line in self.get_texts():
            yield self.dictionary.doc2bow(self.tokenize_line(line))
    
    def __len__(self):
        return sum(1 for doc in self)  
        
    def __getitem__(self, key):
        # key is line number in the file
        for i, line in enumerate(self.get_texts()):
            if i == key:
                return line.strip()

        raise KeyError
            

def create_corpus_and_dictionary(file):
    print "Creating corpus and dictionary from <%s>." % file
    corpus = IterativeFileCorpus(file)
    return corpus, corpus.dictionary

def create_model(corpus, num_topics):
    return models.LsiModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
    
def run_query(corpus, dictionary, model, match_corpus, query_string):
    """Evaluate a query against possible matches given a:
    Corpus
    Dictionary
    Model
    
    match_corpus is a corpus against which to compare the query string.
    query_string is a string.
    
    """

    query_vec_bow = dictionary.doc2bow(query_string.lower().split())
    query_vec_lsi = model[query_vec_bow] # convert the query to LSI space, same space as the model the corpus went into
    # query_vec_lsi is the coordinates of the query in LSI space

    # create the index of the match corpus using the already built model
    index = similarities.MatrixSimilarity(model[match_corpus])
    
    sims = index[query_vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # pretty_results = tuple((fcb.corpus_lines[sim[0]], sim[1]) for sim in sims)
    # results are (string, score) tuples
    pretty_results = tuple((match_corpus[sim[0]], float(sim[1])) for sim in sims)

    for i, pretty_result in enumerate(pretty_results):
        tw = textwrap.fill("{0:d}. ({1: 0.2g}) '{2:s}'".format(i+1, pretty_result[1], pretty_result[0]), initial_indent="  ", subsequent_indent="  ")
        print tw


def loop_query(corpus, dictionary, model, possible_matches=None, query=None):
    while True:
        if not query:
            print "> ",
            query = sys.stdin.readline().strip()

        query_lower = query.lower()
        
        if query_lower in DONE_WORDS:
            break
        elif query_lower.startswith("topics "):
            pass
        elif query_lower.startswith("distance "):
            cmd = query.split()
            if len(cmd) == 3:
                match_against = IterativeFileCorpus([cmd[2]], dictionary=dictionary)
                run_query(corpus, dictionary, model, match_against, cmd[1])
        elif possible_matches:
            run_query(corpus, dictionary, model, possible_matches, query)

        query =  None
        
def save(corpus, dictionary, model, corpus_filename, dict_filename, model_filename):
    print "Saving %s, %s, and %s." % (corpus_filename, dict_filename, model_filename)
    corpora.MmCorpus.serialize(corpus_filename, corpus) # store to disk, for later use
    dictionary.save(dict_filename) # store the dictionary, for future reference
    model.save(model_filename) # same for tfidf, lda, ...
    
def load(corpus_file, dict_file, model_file):
    print "Loading %s, %s, and %s." % (corpus_file, dict_file, model_file)
    dictionary = corpora.Dictionary.load(dict_file)
    corpus = corpora.MmCorpus(corpus_file)
    lsi = models.LsiModel.load(model_file)
    
    return corpus, dictionary, lsi

def print_usage():
    print "Usage:"
    print "astronaut.py generate <corpus_text_file> <num_topics> <corpus_out_file> <dict_out_file> <model_out_file>"
    print "astronaut.py load <corpus_file> <dict_file> <model_file> [candidate_corpus_file]"
    
def load_possible_matches(file):
    with open(file) as f:
        possible_matches = f.readlines()
    return [pm.strip() for pm in possible_matches]
    
def main():
    if len(sys.argv) < 4:
        print_usage()
        return

    if sys.argv[1] == "generate":
        if len(sys.argv) == 7:
            corpus, dictionary = create_corpus_and_dictionary(sys.argv[2])
            model = create_model(corpus, sys.argv[3])
            save(corpus, dictionary, model, sys.argv[4], sys.argv[5], sys.argv[6])

    elif sys.argv[1] == "load":
        if len(sys.argv) < 5:
            print_usage()
            return
        else:
            corpus, dictionary, model = load(sys.argv[2], sys.argv[3], sys.argv[4])
            possible_matches_corpus = None
            if len(sys.argv) >= 6:
                possible_matches_file = sys.argv[5]
                possible_matches_corpus = IterativeFileCorpus(possible_matches_file, dictionary)
        
            loop_query(corpus, dictionary, model, possible_matches_corpus)

    else:
        print_usage()


if __name__ == "__main__":
    main()

