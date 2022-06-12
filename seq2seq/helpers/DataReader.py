from helpers.PreProcessing import NormalizeString
from helpers.Lang import Lang
from models.config_WOAttention import *

class DataReader:
    """
     Given a language pair, DataReader
     (i) reads and splits data from a text file comprising "src trg"
     (ii) normalizes, trims the sentences and creates a list from them
     (iii) returns input_lang, output_lang, pairs
     which input language and output language are the objects of the "Lang" class.
    """
    def __init__(self, lang1, lang2, max_length = 10, reverse = False):
        self.lang1 = lang1
        self.lang2 = lang2
        self.max_length = max_length
        self.reverse = reverse

    def ReadLines(self):
        print("Reading lines...")
        lines = open("data/%s-%s.txt" % (self.lang1, self.lang2), encoding="utf-8").read().strip().split("\n")
        return lines

    def Splitter(self):
        lines = self.ReadLines()
        pairs = [[NormalizeString(s) for s in l.split('\t')] for l in lines]
        print("Read %s sentence pairs" % len(pairs))
        pairs = [pair for pair in pairs if self.filterPair(pair)]
        return pairs

    def filterPair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
                   len(p[1].split(' ')) < self.max_length

    def Read(self):
        pairs = self.Splitter()

        if self.reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.lang2)
            output_lang = Lang(self.lang1)
        else:
            input_lang = Lang(self.lang1)
            output_lang = Lang(self.lang2)

        return input_lang, output_lang, pairs

    def PrepareData(self):
        """
        PrepareData takes input and output sentences as pairs
        e.g., ['la chambre est trop petite .', 'the room is too small .']
        then passes the src-side and trg-side to the Lang class to create indices.
        """
        input_lang, output_lang, pairs = self.Read()
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting words...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])

        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)

        return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(" ")]

def tensorFromSentence(lang, sentence, device):
    indexes  = indexesFromSentence(lang, sentence)
    indexes.append(config_WOAttention['EOS_token'])
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang, device):
    input_tensor = tensorFromSentence(input_lang, pair[0], device)
    target_tensor = tensorFromSentence(output_lang, pair[1], device)
    return (input_tensor, target_tensor)
