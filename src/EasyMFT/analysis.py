import numpy as np
import pandas as pd

def analyze(document_iterator, mfd):
    ids = []
    titles = []
    corpus_morals = []
    word_counts = []
    foundations = set([])
    
    for doc in document_iterator:
        docid = doc['id']
        title = doc['title']
        text = doc['text']
        words = text.split()
        
        document_morals = {}
        for word in words:
            morals = mfd.score(word)
            for m in morals:
                foundations.add(m)
                if m not in document_morals:
                    document_morals[m] = 0
                document_morals[m] += morals[m]
                
        ids.append(docid)
        titles.append(title)
        corpus_morals.append(document_morals)
        word_counts.append(len(words))
            
    # organize the foundations
    foundations = list(foundations)
    id2foundation = foundations
    foundation2id = {f:i for i,f in enumerate(foundations)}
    foundation_mtx = np.zeros((len(ids), len(foundations)))
    for i, doc_morals in enumerate(corpus_morals):
        for f in doc_morals:
            foundation_mtx[i, foundation2id[f]] = doc_morals[f]
          
    # Build the dataframe
    data = {
        'id':ids,
        'title':titles,
        'word_count':word_counts,
    }
    for i,f in enumerate(foundations):
        data[f] = foundation_mtx[:, i]
        
    df = pd.DataFrame(data)
    return df