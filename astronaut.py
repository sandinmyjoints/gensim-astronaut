#!/usr/bin/env python
# -*- coding: utf-8 -*-
from cmd import Cmd

import string
import sys
import argparse
import textwrap
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models, similarities, interfaces


PUNCT_CHARS = "!@#$%^&*`,.:;'\"[](){}"

def make_trans_table(trans_from, trans_to, remove=""):
    trans_tab = dict(zip(map(ord, trans_from), map(ord, trans_to)))
    trans_tab.update((ord(c), None) for c in remove)
    return trans_tab

PUNCT_TRANS_TABLE = make_trans_table('-', " ", remove=PUNCT_CHARS)

def make_tokenize_line_func(trans_table=PUNCT_TRANS_TABLE):
    def tokenize_line(line, trans_table=trans_table):
        line = unicode(line, encoding="iso8859").translate(trans_table)
        line = line.lower().split()
        return line
    return tokenize_line

default_tokenize_line = make_tokenize_line_func()

def getstream(input):
    """
    If input is a filename (string), return `open(input)`.
    If input is a file-like object, reset it to the beginning with `input.seek(0)`.
    If input is a list, tuple, etc., pass it on as result.
    """
    assert input is not None
    if isinstance(input, basestring):
        # input was a filename: open as text file
        result = open(input)
    else:
        # input was a file-like object (BZ2, Gzip etc.); reset the stream to its beginning
        result = input

        if hasattr(result, "seek"):
            result.seek(0)

    return result

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
        for line in getstream(self.input):
            yield self.tokenize_line(line)

    def ibuild_dictionary(self):
        dictionary = corpora.Dictionary(text for text in self.get_texts())

        # remove stop words and words that appear only once
        stop_ids = [dictionary.token2id[stopword] for stopword in self.stoplist
                    if stopword in dictionary.token2id]
        once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq == 1]
        dictionary.filter_tokens(stop_ids + once_ids) # remove stop words and words that appear only once
        dictionary.compactify() # remove gaps in id sequence after words that were removed        

        print "%d terms in dictionary." % len(dictionary)
        return dictionary
            
    def __iter__(self):
        for text in self.get_texts():
            yield self.dictionary.doc2bow(text)
    
    def __len__(self):
        return sum(1 for doc in self)  
        
    def __getitem__(self, key):
        # return the original line from the input
        # key is line number in the file/index in the list/etc
        for i, line in enumerate(getstream(self.input)):
            if i == key:
                return line.strip()

        raise KeyError
            

def create_corpus_and_dictionary(file):
    print "Creating corpus and dictionary from <%s>." % file
    corpus = OptionalDictCorpus(file)
    return corpus, corpus.dictionary

def create_model(corpus, num_topics):
    return models.LsiModel(corpus, id2word=corpus.dictionary, num_topics=num_topics)
    
def run_query(corpus, dictionary, model, match_corpus, query_string):
    """Evaluate a query against possible matches given a:
    Corpus
    Dictionary
    Model
    
    match_corpus is an OptionalDictCorpus against which to compare the query string.
    query_string is a string.
    
    """

    query_vec_bow = dictionary.doc2bow(match_corpus.tokenize_line(query_string))
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
        tw = textwrap.fill("{0:d}. ({1: 0.2f}) '{2:s}'".format(i+1, pretty_result[1], pretty_result[0]), initial_indent="  ", subsequent_indent="  ")
        print tw


class AstronautCmd(Cmd):

    def __init__(self, corpus, dictionary, model, possible_matches=None):
        Cmd.__init__(self)
        self.prompt = "> "
        self.corpus = corpus
        self.dictionary = dictionary
        self.model = model
        self.possible_matches = possible_matches

    def do_vector(self, command):
        pass

    def do_topics(self, command):
       cmd = command.split()
       if len(cmd) == 2:
           try:
               for i, topic in enumerate(self.model.show_topics(num_topics=int(cmd[0]), num_words=int(cmd[1]))):
                   print "Topic {0:d}. ({1})".format(i, topic)
               return
           except Exception, ex:
#                   print ex
               pass

       print "Usage: topics <num_topics> <num_words_per_topic>"

    def do_distance(self, command):
        cmd = command.split()
        if len(cmd) == 2:
            try:
                match_against = OptionalDictCorpus([cmd[1]], dictionary=self.dictionary)
                run_query(self.corpus, self.dictionary, self.model, match_against, cmd[0])
                return
            except Exception, ex:
                print ex
                pass

        print "Usage: distance <term1> <term2>"

    def default(self, command):
        if command in DONE_WORDS:
            return True

        run_query(self.corpus, self.dictionary, self.model, self.possible_matches, command)

    def do_exit(self, command):
        do_EOF(command)

    def do_quit(self, command):
        do_EOF(command)

    def do_EOF(self, command):
        return True

def save(corpus, dictionary, model, corpus_filename, dict_filename, model_filename):
    print "Saving %s, %s, and %s." % (corpus_filename, dict_filename, model_filename)
    corpora.MmCorpus.serialize(corpus_filename, corpus) # store to disk, for later use
    dictionary.save(dict_filename) # store the dictionary, for future reference
    model.save(model_filename) # same for tfidf, lda, ...
    
def load(corpus_file, dict_file, model_file, dictfromtext=False):
    print "Loading %s, %s, and %s." % (corpus_file, dict_file, model_file)
    if dictfromtext:
        dictionary = corpora.Dictionary.load_from_text(dict_file)
    else:
        dictionary = corpora.Dictionary.load(dict_file)
    corpus = corpora.MmCorpus(corpus_file)
    model = models.LsiModel.load(model_file)
    
    return corpus, dictionary, model

def print_usage():
    print "Usage:"
    print "astronaut.py generate <corpus_text_file> <num_topics> <corpus_out_file> <dict_out_file> <model_out_file>"
    print "astronaut.py load <corpus_file> <dict_file> <model_file> [candidate_corpus_file]"
        
def main():
    if len(sys.argv) < 4:
        print_usage()
        return

    if sys.argv[1] == "generate":
        if len(sys.argv) == 7:
            corpus, dictionary = create_corpus_and_dictionary(sys.argv[2])
            model = create_model(corpus, sys.argv[3])
            save(corpus, dictionary, model, sys.argv[4], sys.argv[5], sys.argv[6])
            loop_query(corpus, dictionary, model)

    elif sys.argv[1] == "load":
        if len(sys.argv) < 5:
            print_usage()
            return
        else:
            corpus, dictionary, model = load(sys.argv[2], sys.argv[3], sys.argv[4])
            possible_matches_corpus = None
            if len(sys.argv) >= 6:
                possible_matches_file = sys.argv[5]
                possible_matches_corpus = OptionalDictCorpus(possible_matches_file, dictionary=dictionary)
        
#            loop_query(corpus, dictionary, model, possible_matches_corpus)
            astro_cmd = AstronautCmd(corpus, dictionary, model, possible_matches_corpus)
            astro_cmd.cmdloop()

    else:
        print_usage()


if __name__ == "__main__":
    main()

