import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def vectorize_using_scikit_learn(texts):
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    vectorizer.fit(texts)
    x_train = vectorizer.transform([texts[0]])

    print 'Vocabulary word'
    print vectorizer.vocabulary_

    print 'CountVectorizer binary word'
    print x_train

    # vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    # vectorizer.fit(texts)
    # x_train = vectorizer.transform([texts[0]])

    print 'Vocabulary char'
    print vectorizer.vocabulary_

    print 'Transforming text'
    print texts[0]

    # print 'CountVectorizer binary char'
    # print x_train

    print 'CountVectorizer count'
    vectorizer = CountVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    vectorizer.fit(texts)
    x_train = vectorizer.transform([texts[0]])
    print x_train

    # print 'CountVectorizer count char'
    # vectorizer = CountVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    # vectorizer.fit(texts)
    # x_train = vectorizer.transform([texts[0]])
    # print x_train

    print 'TfidfVectorizer'
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    vectorizer.fit(texts)
    x_train = vectorizer.transform([texts[0]])
    print x_train

    # print 'TfidfVectorizer char'
    # vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    # vectorizer.fit(texts)
    # x_train = vectorizer.transform([texts[0]])
    # print x_train

    print 'HashingVectorizer'
    vectorizer = HashingVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', n_features=25)
    vectorizer.fit(texts)
    x_train = vectorizer.transform([texts[0]])
    print x_train

    # print 'HashingVectorizer char'
    # vectorizer = HashingVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char', n_features=25)
    # vectorizer.fit(texts)
    # x_train = vectorizer.transform([texts[0]])
    # print x_train


if __name__ == '__main__':
    data = ['John likes to watch movies. Mary likes movies too.',
            'John also likes to watch football games.']

    vectorize_using_scikit_learn(data)
