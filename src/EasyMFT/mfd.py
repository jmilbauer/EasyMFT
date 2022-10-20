"""
Contains code for downloading, processing, and using the moral foundations dictionary.
"""

import abc
from abc import ABC
import json
from sklearn.metrics.pairwise import cosine_similarity

class MoralFoundationsDictionary(ABC):
    """A moral foundations dictionary.
    
    The dictionary has two types of analysis:
        1) Strict - Uses only the original Stem->Foundation mapping from:
        2) Word2Vec - Given a word embedding matrix, uses the technique described by:
    """
    
    @abc.abstractmethod
    def initialize(self):
        pass
    
    @abc.abstractmethod
    def search(self, word):
        pass
    
    def score(self, word):
        moral_scores = self.search(word)
        z = sum(moral_scores.values())
        for m in moral_scores:
            moral_scores[m] /= z
        return moral_scores

class MoralFoundationsDictionary_STRICT(MoralFoundationsDictionary):
    
    def __init__(self, mfd_path):
        self.mfd_path = mfd_path
        
    def _build_trie(self, mfd_dict):
        root = dict()
        for word in mfd_dict:
            current_dict = root
            for letter in word:
                current_dict = current_dict.setdefault(letter, {})
            current_dict[self._terminal] = mfd_dict[word]
        return root
    
    def _search_trie(self, trie, word):
        current_dict = trie
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            elif "*" in current_dict:
                current_dict = current_dict['*']
                break
            else:
                return {}
        while '*' in current_dict:
            current_dict = current_dict['*']
        return current_dict
    
    def initialize(self):
        self._terminal = '_end_'
        self.mfd_dict = json.load(open(self.mfd_path))
        self.trie = self._build_trie(self.mfd_dict)
        
    def search(self, word):
        search_result = self._search_trie(self.trie, word)
        if self._terminal in search_result:
            return {k:1 for k in search_result[self._terminal]}
        else:
            return {}
        
        
class MoralFoundationsDictionary_W2V(MoralFoundationsDictionary):
    
    def __init__(self, mfd_path, w2v_dict):
        self.mfd = MoralFoundationsDictionary_STRICT(mfd_path)
        self.word2vec = w2v_dict
        self.vocabulary = list(w2v_dict.keys())
        
    def initialize(self):
        self.mfd.initialize()
        
        # get the key words for each moral foundation.
        foundation2words = {}
        for w in self.vocabulary:
            foundations = self.mfd.search(w)
            for f in foundations:
                if f not in foundation2words:
                    foundation2words[f] = []
                foundation2words[f].append(w)
                
        # find the w2v centroid for each moral foundation
        foundation2centroid = {}
        for f in foundation2words:
            foundation_words = foundation2words[f]
            foundation_vectors = [self.word2vec[w] for w in foundation_words]
            mtx = np.vstack(foundation_vectors)
            centroid = np.mean(mtx, axis=0)
            foundation2centroid[f] = centroid
            
        # build the foundation matrix
        foundations, foundation_mtx = list(zip(*foundation2centroid.items()))
        foundation_mtx = np.vstack(foundation_mtx)
        
        # build the similarity matrix
        word_mtx = np.vstack([self.word2vec[w] for w in self.vocabulary])
        similarity_matrix = cosine_similarity(word_mtx, foundation_mtx)
        
        # store what is needed
        self.foundations = foundations
        self.similarity = similarity_matrix
        self.word2id = {w:i for i,w in enumerate(self.vocabulary)}
        
    def search(self, word):
        if word not in self.word2id:
            return {}
        vec = self.similarity[self.word2id[word], :]
        res = {}
        for i, score in enumerate(vec):
            res[self.foundations[i]] = score
        return res
    
MoralFoundationsDictionary.register(MoralFoundationsDictionary_STRICT)
MoralFoundationsDictionary.register(MoralFoundationsDictionary_W2V)