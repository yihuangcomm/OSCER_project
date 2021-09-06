import nltk
import sklearn_crfsuite

# split sentences from original data.
def make_sents(data):
    sents = {}
    i = 0
    j = 0
    sents[j] = []
    while i<data.shape[0]:               
        sents[j].append(data[i])
        if data[i][0]=='.' or data[i][0]=='?':  # Used '.' and '?' as the sentence indicators, because there are not '!' and '...'.  
             j = j+1 
             sents[j] = []
        i = i+1
    return sents
                   

# word to features
def word2features(sent, i):
    word = sent[i][0]
    label = sent[i][1]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),     
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
        
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [sent[i][1] for i in range(len(sent))]

def sent2tokens(sent):
    return [sent[i][0] for i in range(len(sent))]
    
# output entity
def output_entity(tokens, labels):
    Entities = []
    Entity = ''
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j]== 'B':
                 if len(Entity)>0: 
                     Entities.append(Entity)
                     Entity = ''
                     Entity = Entity + tokens[i][j]
                 else:
                     Entity = Entity + tokens[i][j]
            elif labels[i][j]== 'I' and len(Entity)>0:
                 Entity = Entity + ' ' + tokens[i][j]
    return Entities
