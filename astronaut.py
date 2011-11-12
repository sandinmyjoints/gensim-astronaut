import sys
import argparse
from scipy.spatial.distance import cdist
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities, interfaces


DONE_WORDS = ["done", "exit", "quit", "q"]
PUNCT_CHARS = ",.:;-'\"[]()"


class OptionalDictCorpus(interfaces.CorpusABC):
    """A corpus that can create its own dictionary or have another dictionary specified,
    which is useful when you want to compare queries against a corpus in an already
    computed space.
    """
    
    def __init__(self, file, dictionary=None):
        self.stoplist = set(stopwords.words('english'))
        self.file = file
        if dictionary:
            self.dictionary = dictionary
        else:
            self.dictionary = self.ibuild_dictionary(file)
        # self.corpus = None
        # self.corpus_lines = None
        len_self = len(self)
        print "%d lines in corpus." % len_self
        if len_self < 40:
            print enumerate(self)
        
    def ibuild_dictionary(self, file):
        print "Building dictionary from <%s>. " % file, 
        dictionary = corpora.Dictionary(line.lower().translate(None, PUNCT_CHARS).split() for line in open(file))
        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify() # remove gaps in id sequence after words that were removed        
        print "%d terms in dictionary." % len(dictionary)
        if len(dictionary) < 40:
            print dictionary
        return dictionary
            
    def __iter__(self):
        for line in open(self.file):
            yield self.dictionary.doc2bow(line.lower().split())
    
    def __len__(self):
        return sum(1 for doc in self)  
        
    def __getitem__(self, key):
        # key is line number in the file
        with open(self.file) as f:
            for i, line in enumerate(f):
                if i == key:
                    return line.strip()

        raise KeyError
            

def create_corpus_and_dictionary(file):
    print "Creating corpus."    
    corpus = IterativeFileCorpus(file)
    return corpus, corpus.dictionary

def create_model(corpus, dims):
    return models.LsiModel(corpus, id2word=corpus.dictionary, num_topics=dims)
    
def run_query(corpus, dictionary, model, match_corpus, query_string):
    """Evaluate a query against possible matches given a:
    Corpus
    Dictionary
    Model
    
    match_corpus is a corpus against which to compare the query string.
    query_string is a string.
    
    """
    
    print " Querying on '%s'." % query_string
    # print " Possible matches are %s" % [(match_corpus[i], v) for i, v in enumerate(match_corpus)]
    print

    query_vec_bow = dictionary.doc2bow(query_string.lower().split())
    query_vec_lsi = model[query_vec_bow] # convert the query to LSI space, same space as the model the corpus went into
    # query_vec_lsi is the coordinates of the query in LSI space

    # create the index of the match corpus using the already built model
    index = similarities.MatrixSimilarity(model[match_corpus])
    
    sims = index[query_vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # pretty_results = tuple((fcb.corpus_lines[sim[0]], sim[1]) for sim in sims)
    pretty_results = tuple((match_corpus[sim[0]], sim[1]) for sim in sims)

    for i, pretty_result in enumerate(pretty_results):
        print "%d. (% 0.2f) '%s'" % (i, pretty_result[1], pretty_result[0])


def loop_query_against_corpus(corpus, dictionary, model, possible_matches, query=None):
    while True:
        if not query:
            print "> ",
            query = sys.stdin.readline().strip()
        
        if query.lower() in DONE_WORDS:
            return
        
        run_query(corpus, dictionary, model, possible_matches, query)
        
def save(corpus, dictionary, model):
    print "Saving %s, %s, and %s." % (corpus, dictionary, model)
    corpora.MmCorpus.serialize('/tmp/abc_corpus500.mm', corpus) # store to disk, for later use
    dictionary.save('/tmp/abc_dictionary500.dict') # store the dictionary, for future reference
    model.save('/tmp/abc_model500.lsi') # same for tfidf, lda, ...
    
def load(corpus_file, dict_file, model_file):
    print "Loading %s, %s, and %s." % (corpus_file, dict_file, model_file)
    dictionary = corpora.Dictionary.load(dict_file)
    corpus = corpora.MmCorpus(corpus_file)
    lsi = models.LsiModel.load(model_file)
    
    return corpus, dictionary, lsi

def print_usage():
    print "Usage:"
    print "astronaut.py generate <corpus_text_file> <num_topics> <corpus_out_file> <dict_out_file> <model_out_file>"
    print "astronaut.py load <corpus_file> <dict_file> <model_file> <possible_matches_file>"
    
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
        if len(sys.argv) < 6:
            print_usage()
            return
        else:
            corpus, dictionary, model = load(sys.argv[2], sys.argv[3], sys.argv[4])
            possible_matches_file = sys.argv[5]
    
    possible_matches_corpus = IterativeFileCorpus(possible_matches_file, dictionary)
        
    loop_query_against_corpus(corpus, dictionary, model, possible_matches_corpus)


if __name__ == "__main__":
    main()

