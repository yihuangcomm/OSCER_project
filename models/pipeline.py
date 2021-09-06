import gflags
from train import main
from test import evaluation
import pandas as pd
import numpy as np
import sys
from data_processing import make_sents, sent2features, sent2labels, sent2tokens, output_entity
from sklearn_crfsuite import metrics

if __name__ == '__main__':

    Flags = gflags.FLAGS
    gflags.DEFINE_string("alg", "lbfgs", "lbfgs, l2sgd ,ap, pa, arow")
    gflags.DEFINE_float("l1", 0.1, "the coefficient for L1 regularization. Supported training algorithms: lbfgs")  
    gflags.DEFINE_float("l2", 0.1, "the coefficient for L2 regularization.Supported training algorithms: l2sgd, lbfgs")
    gflags.DEFINE_integer("max_iter", 1000, "number of iterations before stopping")
    gflags.DEFINE_bool('is_all_states', False, "whether CRFsuite generates state features that do not even occur in the training data") 
    gflags.DEFINE_bool('is_all_transitions', True, "whether CRFsuite generates transition features that do not even occur in the training data") 
    gflags.DEFINE_string("train_path", "../NERdata/train.tsv", "path of training file")
    gflags.DEFINE_string("test_path", "../NERdata/test.tsv", 'path of testing file')       
    Flags(sys.argv)

#Loading trainig and test data

train_data = pd.read_csv (Flags.train_path, sep = '\t', engine='python', header=None).loc[:]
test_data = pd.read_csv (Flags.test_path, sep = '\t', engine='python', header=None).loc[:]

train_data = train_data.fillna(method='ffill')
train_data = np.array(train_data)

test_data = test_data.fillna(method='ffill')
test_data = np.array(test_data)
    
train_sents = make_sents(train_data)
test_sents = make_sents(test_data)

X_train = [sent2features(train_sents[i]) for i in range(len(train_sents))]
X_test = [sent2features(test_sents[i]) for i in range(len(test_sents))]
    
y_train = [sent2labels(train_sents[i]) for i in range(len(train_sents))]
y_test = [sent2labels(test_sents[i]) for i in range(len(test_sents))]

token_test = [sent2tokens(test_sents[i]) for i in range(len(test_sents))]

# hyperparameters optimization

L_list = list(range(11))
F1_scores = []
best_f1_score = 0.0
best_l1 = 0.0
best_l2 = 0.0

for i in L_list:
    for j in L_list:
        ner_model, labels = main(train_features=X_train, train_labels=y_train, alg=Flags.alg, l1=0.1*i, l2=0.1*j, max_iter=100)
        f1_score, _ = evaluation(model=ner_model, test_features=X_test, test_labels=y_test, labels=labels)
        F1_scores.append(f1_score)
        if f1_score>best_f1_score:
            best_f1_score = f1_score
            best_l1 = 0.1*i
            best_l2 = 0.1*j
        print('f1_score:%.4f, current_L1:%f, current_L2:%f, Best_f1_score:%.4f, Best_L1:%f, Best_L2:%f' % (f1_score, 0.1*i, 0.1*j, best_f1_score, best_l1, best_l2))


# Fit the best model
best_model, labels = main(train_features=X_train, train_labels=y_train, alg=Flags.alg, l1=best_l1, l2=best_l2, max_iter=1000)

# Evaluate the performance and output the result
f1_score, y_pred = evaluation(model=best_model, test_features=X_test, test_labels=y_test, labels=labels)

Entities = output_entity(token_test, y_pred)
Entities = set(Entities)
print('Entities:%s' % (str(Entities)))
print('Number of Entities: %d' % (len(Entities)))

print('Weighted Average F1 Score: %.4f' % (f1_score))  
                    
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
print(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
    
with open('result', 'a') as f:
    f.write('Number of Entities'+str(len(set(Entities)))+'\n')
    f.write('Entities:'+str(set(Entities))+'\n')
    f.write('Weighted Average F1 Score:'+str(f1_score)+'\n')
    f.write(str(metrics.flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))+'\n')
