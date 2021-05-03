
import os
import json
from collections import OrderedDict,Counter


class Vocab():


    def __init__(self):

        self.word_to_idx = dict()
        # Maintains the number of words in the vocabulary
        self.size = 0
        # Maintains the total number of indices in the vocabulary including the offset
        self.total_vocab_size = 0
        # Maintains the last index in the vocabulary
        self.index = 0
        # Maintains the first index from which the vocabulary should start
        self.offset = 0
        # End of the line marker which will be substituted for \n in the text
        self.eol_marker = " "           


    def buildFromFolder(self, folder:str, vocab_size:int, offset:int, tokenizer:object):
        """  
        @params:
        --------
        - folder:       path to the vocabulary folder containing .txt files
        - vocab_size:   number of words desired in the vocabulary
        - offset:       indexing offset for the vocabulary
        - tokenizer:    python function which takes a string as input and tokenizes it 
                        into a list of sub strings according to the tokenization level
        """
        contents = ""

        # Read individual text file and call the buildFromText)() function
        paths = [os.path.join(folder,fname) for fname in os.listdir(folder) if fname.endswith(".txt")]
        for path in paths:
            with open(path, "r") as fp:
                text = fp.read().lower().replace("\n", self.eol_marker)
                
            contents += self.eol_marker+text
        self.buildFromText(contents, vocab_size, offset, tokenizer)


    def buildFromText(self, text:str, vocab_size:int, offset:int, tokenizer:object):
        """  
        @params:
        --------
        - text:         raw text from the .txt file
        - vocab_size:   number of words desired in the vocabulary
        - offset:       indexing offset for the vocabulary
        - tokenizer:    python function which takes a string as input and tokenizes it 
                        into a list of sub strings according to the tokenization level
        """
        self.size = vocab_size
        self.total_vocab_size = self.size  + offset 
        self.index = offset
        self.offset = offset

        tokens = tokenizer(text)    

        # Sort the words in decreasing order of the frequency and keep the top 'vocab_size' words    
        word_freq = OrderedDict(sorted(Counter(tokens).items(), key=lambda x:x[1], reverse=True)[0:vocab_size])
        for word in word_freq.keys():
            if self.word_to_idx.get(word, None) is None:
                self.word_to_idx[word] = self.index
                self.index += 1


    def buildFromFile(self, path:str, vocab_size:int, offset:int, tokenizer:object):
        """  
        @params:
        --------
        - path:         path to the .txt file
        - vocab_size:   number of words desired in the vocabulary
        - offset:       indexing offset for the vocabulary
        - tokenizer:    python function which takes a string as input and tokenizes it 
                        into a list of sub strings according to the tokenization level
        """
        with open(path, "r") as fp:
            text = fp.read().lower().replace("\n", self.eol_marker)

        self.buildFromText(text, vocab_size, offset, tokenizer)

    
    def load(self, path:str):
        """  
        @params:
        --------
        - path: path to the JSON file containing the vocab
        """
        with open(path, "r") as fp:
            vocab = json.load(fp)

        self.word_to_idx = vocab
        vocab_len = len(vocab)

        self.offset = self.word_to_idx[min(self.word_to_idx, key=self.word_to_idx.get)]

        self.index = vocab_len + self.offset
        self.size = vocab_len
        self.total_vocab_size = vocab_len + self.offset


    def save(self, path:str):
        """  
        @params:
        --------
        - path: path to the JSON file holding the vocab
        """
        with open(path, "w") as fp:
            json.dump(self.word_to_idx, fp)
        
