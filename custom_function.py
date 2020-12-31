#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chương trình chuyển đổi từ Tiếng Việt có dấu sang Tiếng Việt không dấu
"""

import re


def no_accent_vietnamese(s):
    s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
    s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
    s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
    s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
    s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
    s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
    s = re.sub(r'[ìíịỉĩ]', 'i', s)
    s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
    s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
    s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
    s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
    s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
    s = re.sub(r'[Đ]', 'D', s)
    s = re.sub(r'[đ]', 'd', s)
    return s
def email_regex_search(st):
    addresses = re.findall(r'[\w\.-]+@[\w\.-]+', doc)
    return addresses
def replace_email_with_regex(email_replace,st):
    new_email_address = re.sub(r'([\w\.-]+)@([\w\.-]+)',r'pqr@mno.com', doc)
    return new_email_address
def text_std(input_text):
    words = input_text.split()
    new_words = []
    for word in words:
        word = re.sub(r'[^\w\s]',",word)
        if word.lower() in lookup_dict:
            word = lookup_dict[word.lower()]
            new_words.append(word)
            new_text = " ".join(new_words)
    return new_text
def work_cloud_generate(data,work_cloud_generate_type=1):
# if work_cloud_generate_type=2. Generate work cloud from frequency dictionary data will be dictionary frequency, else = 1 from text data will be text
    from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
    import matplotlib.pyplot as plt
    if work_cloud_generate_type==1:
        wordcloud = WordCloud().generate(data)
    else:
        wordcloud = WordCloud().generate_from_frequencies(data)
    plt.figure(figsize=(18,14))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
"""
# Work Embeddings

sentences = [['I', 'love', 'nlp'],
['I', 'will', 'learn', 'nlp', 'in', '2','months'],
['nlp', 'is', 'future'],
['nlp', 'saves', 'time', 'and', 'solves',
'lot', 'of', 'industry', 'problems'],
['nlp', 'uses', 'machine', 'learning']]


# Skip grams
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
skipgram = Word2Vec(sentences, size =50, window = 3, min_count=1,sg = 1)

skipgram.save('skipgram.bin')
skipgram = Word2Vec.load('skipgram.bin')

# visualize T – SNE plot
X = skipgram[skipgram.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
pyplot.scatter(result[:, 0], result[:, 1])
words = list(skipgram.wv.vocab)
    for i, word in enumerate(words):
pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()



#CBOW

Gần giống với skip-gram
skip gram từ vựng đã được google train cho tiếng anh
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

#fastText
from gensim.models import FastText
from sklearn.decomposition import PCA
from matplotlib import pyplot
fast = FastText(sentences,size=20, window=1, min_count=1,workers=5, min_n=1, max_n=2)

X = fast[fast.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

pyplot.scatter(result[:, 0], result[:, 1])
words = list(fast.wv.vocab)
    for i, word in enumerate(words):
pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()




"""
