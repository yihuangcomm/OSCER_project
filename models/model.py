import sklearn_crfsuite

def NER_crf(alg, l1,l2,max_iter, is_all_states, is_all_transitions):
    ner_crf = sklearn_crfsuite.CRF(
        algorithm=alg,
        c1=l1,
        c2=l2,
        max_iterations=max_iter,
        all_possible_states=is_all_states,
        all_possible_transitions=is_all_transitions
    )
    return ner_crf
