"""BM25 written by hand with an inverted index (issue #19).

Import from a notebook in this folder with `from bm25 import BM25, tokenize`.
Stays here, not in `src/ranking/`, until issue #52.

Formula (the same as rank_bm25's BM25Okapi, so it can be checked against it)
    idf(t)   = log(N - n_t + 0.5) - log(n_t + 0.5)      n_t = documents containing t
               if idf < 0 it is replaced by epsilon * average idf
    score    = sum over query tokens t of idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
"""

import math
import re
from collections import Counter

import numpy as np


def tokenize(text):
    """Lowercase and split on anything that is not a letter or digit."""
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.n = len(docs)
        self.doc_len = np.array([len(d) for d in docs], dtype=float)
        self.avgdl = self.doc_len.mean()
        postings = {}
        for i, doc in enumerate(docs):
            for token, tf in Counter(doc).items():
                postings.setdefault(token, ([], []))
                postings[token][0].append(i)
                postings[token][1].append(tf)
        # inverted index: token -> (array of doc ids, array of term frequencies)
        self.index = {t: (np.array(ids), np.array(tfs, dtype=float)) for t, (ids, tfs) in postings.items()}
        idf = {t: math.log(self.n - len(ids) + 0.5) - math.log(len(ids) + 0.5) for t, (ids, _) in self.index.items()}
        average = sum(idf.values()) / len(idf)
        self.idf = {t: (v if v >= 0 else epsilon * average) for t, v in idf.items()}

    def scores(self, query_tokens):
        """BM25 score of every document for a query (a list of tokens). Zero if no token matches."""
        out = np.zeros(self.n)
        for token in query_tokens:
            if token not in self.index:
                continue
            ids, tf = self.index[token]
            norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[ids] / self.avgdl)
            out[ids] += self.idf[token] * tf * (self.k1 + 1) / norm
        return out
