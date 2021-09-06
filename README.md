# OSCER_project
A NER(Named-entity recognition)  model to detect the entities from the given texts.
## Requirements
Experiments were implemented on the high performance computing (HPC) Platform of a University.

Experiments enviroments settings are as below:

Software:

- Python 3.8.8 
- nltk              3.6.2
- numpy             1.20.1
- pandas            1.2.4
- python-crfsuite   0.9.7
- python-gflags     3.1.2
- scikit-learn      0.24.1
- scipy             1.6.2
- sklearn-crfsuite  0.3.6

OS: Red Hat Enterprise Linux Server release 7.9 

Hardware:
- CPU number: 1
- CPU type: AMD64
- CPU RAM: 4000MB

## Data
All data is in /NERdata. The entities are tagged in BIO format.

test.tsv: dataset to evaluate the model

train.tsv: dataset to train the model

## Run step
- python3 pipeline.py  


