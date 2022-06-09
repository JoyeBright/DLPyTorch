from modules.lib import *

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2 # count SOS and EOS -> 0, 1 Thus, the starting index is 2

    def addSentence(self, sentence):
        """
        Given a sentence, split it and only take the words
        then pass each word to the AddWord method
        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        Given a word, first check if the word is already in the word2index list
        if not:
            (i) assign an index number for that (it starts from 2)
            (ii) set word2count to 1 (as it appears for the first time)
            (iii) assign the word to its index (index was already defined in step i)
            (iv) increase n_words by one for the next word
        else:
            increase the word2count by one
        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] +=1
