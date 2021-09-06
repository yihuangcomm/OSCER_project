import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def evaluation(model, test_features, test_labels, labels):
    test_pred = model.predict(test_features)

    f1_score = metrics.flat_f1_score(test_labels, test_pred, 
                      average='weighted', labels=labels)

    return f1_score, test_pred
    
