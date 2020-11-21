# coding: utf-8
import lzma
import os
import pickle
import sys

sys.path.append("..")
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi, timer
from dataset import ptb

PKL_PATH = "PPMI.pkl.xz"

wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data("train")

if os.path.exists(PKL_PATH):
    with timer("load PPMI"):
        with lzma.open(PKL_PATH, "rb") as fin:
            W = pickle.load(fin)
else:
    window_size = 2
    vocab_size = len(word_to_id)
    print("counting  co-occurrence ...")
    with timer("calc co-occurrence"):
        C = create_co_matrix(corpus, vocab_size, window_size)
    print("calculating PPMI ...")
    with timer("calc PPMI"):
        W = ppmi(C, verbose=True)
    with lzma.open(PKL_PATH, "wb") as fout:
        pickle.dump(W, fout)

try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd

    with timer("calc SVD with sklean.randomized_svd"):
        U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5, random_state=None)
except ImportError:
    # SVD (slow)
    with timer("calc SVD with np.linalg.svd"):
        U, S, V = np.linalg.svd(W)

word_vecs = U[:, :wordvec_size]

querys = ["you", "year", "car", "toyota"]
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
