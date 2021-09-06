import nltk
from model import NER_crf
import pandas as pd
import numpy as np


def main(train_features, train_labels, alg, l1, l2, max_iter=1000, is_all_states=False, is_all_transitions=True):
    crf = NER_crf(alg, l1, l2, max_iter, is_all_states, is_all_transitions)
    crf.fit(train_features, train_labels)
    labels = list(crf.classes_)
    return crf, labels
    
if __name__ == '__main__':

    main()    
