gensim-astronaut
================

gensim-astronaut is a simple tool for exploring (and generating/persisting) topic indexing (LSI, LDA) spaces created with `gensim <http://radimrehurek.com/gensim>`_ through a command line interface.

Usage
=====

- astronaut.py generate <input_corpus_file> <num_topics> <corpus_out_file> <dict_out_file> <model_out_file>
  Generates a corpus, dictionary, and model from the input corpus file (can be text or compressed text).

- astronaut.py load <corpus_file> <dict_file> <model_file> [candidate_corpus_file]
  Loads the corpus, dictionary, and model files, as well as an optional candidate corpus file. 