from django.shortcuts import render, redirect, get_object_or_404
import gensim
import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

from .models import Document
from .forms import DocumentForm

def document_upload(request):
    documents = Document.objects.order_by('-id')
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('home')
    else:
        form = DocumentForm()
    return render(request, 'document_upload.html', {
        'form': form,
        'documents':documents,
    })
def similarity(request, id):
    document = get_object_or_404(Document, id=id)
    file_docs = []
    file2_docs = []
    avg_sims = []
    with open ('media/' + document.document.name) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_docs.append(line)
            
    length_doc1 = len(file_docs)

    gen_docs = [[w.lower() for w in word_tokenize(text)] 
                for text in file_docs]

    dictionary = gensim.corpora.Dictionary(gen_docs)
    corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
    tf_idf = gensim.models.TfidfModel(corpus)
    sims = gensim.similarities.Similarity('workdir/',tf_idf[corpus],
                                        num_features=len(dictionary))

    with open ('media/' + document.document2.name) as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file2_docs.append(line)
            
    for line in file2_docs:
        query_doc = [w.lower() for w in word_tokenize(line)]
        query_doc_bow = dictionary.doc2bow(query_doc)
        query_doc_tf_idf = tf_idf[query_doc_bow]
        print('Comparing Result:', sims[query_doc_tf_idf]) 
        sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
        avg = sum_of_sims / len(file_docs)
        print(f'avg: {sum_of_sims / len(file_docs)}')
        avg_sims.append(avg)  
    #total_avg = np.sum(avg_sims, dtype=np.float)
    #Don't we need to divide the total_avg by the number of averages in the array?
    total_avg = np.sum(avg_sims, dtype=np.float) / len(avg_sims)
    
    print(total_avg)
    percentage_of_similarity = round(float(total_avg) * 100)
    if percentage_of_similarity >= 100:
        percentage_of_similarity = 100

    return render(request, 'document.html', {
        'percentage_of_similarity':percentage_of_similarity,
    })  
