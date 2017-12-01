import sklearn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

def vectorize_using_scikit_learn(texts, val_texts):
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    x_train = vectorizer.fit_transform(texts)

    print 'Vocabulary word'
    print vectorizer.vocabulary_

    print 'CountVectorizer binary word'
    print x_train

    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    x_train = vectorizer.fit_transform(texts)

    print 'Vocabulary char'
    print vectorizer.vocabulary_

    print 'CountVectorizer binary char'
    print x_train

    print 'CountVectorizer count'
    vectorizer = CountVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    x_train = vectorizer.fit_transform(texts)
    print x_train

    print 'CountVectorizer count char'
    vectorizer = CountVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    x_train = vectorizer.fit_transform(texts)
    print x_train

    print 'TfidfVectorizer'
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace')
    x_train = vectorizer.fit_transform(texts)
    print x_train

    print 'TfidfVectorizer char'
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char')
    x_train = vectorizer.fit_transform(texts)
    print x_train

    print 'HashingVectorizer'
    vectorizer = HashingVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', n_features=25)
    x_train = vectorizer.fit_transform(texts)
    print x_train

    print 'HashingVectorizer char'
    vectorizer = HashingVectorizer(ngram_range=(1, 2), dtype='int32', strip_accents='unicode', decode_error='replace', analyzer='char', n_features=25)
    x_train = vectorizer.fit_transform(texts)
    print x_train


if __name__ == '__main__':
    train_data = ['John likes to watch movies. Mary likes movies too.',
            'John also likes to watch football games.']
    test_data = ['John and Mary both like movies.']

    vectorize_using_scikit_learn(train_data, test_data)
