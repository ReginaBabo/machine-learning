import pandas
import tensorflow

from tensorflow.contrib.learn.python.learn.preprocessing import text
from tensorflow.contrib.learn.python.learn.preprocessing import CategoricalVocabulary

def vectorize_using_tensor_flow(texts):
    # Based off of text classification example from tensorflow
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification.py
    # and text tests from
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/preprocessing/tests/text_test.py

    # Process the vocabulary
    MAX_DOCUMENT_LENGTH = 20 # Extra space will be padding
    vocab_processor = tensorflow.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    tokens = vocab_processor.fit_transform(pandas.Series(texts))

    print 'Word id mapping'
    print vocab_processor.vocabulary_._mapping

    print 'Vocabulary processor transformed text'
    print list(tokens)

    print 'Total words:'
    print len(vocab_processor.vocabulary_)

    print 'Tokenizer'
    print list(text.tokenizer(texts))

    print 'Byte processor'
    processor = text.ByteProcessor(MAX_DOCUMENT_LENGTH)
    print list(processor.fit_transform(texts))

    print 'Vocabulary processor transformed text'
    vocab_processor = text.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    tokens = vocab_processor.fit_transform(texts)
    print list(tokens)

    print 'Existing vocabulary processor'
    vocab = CategoricalVocabulary()
    vocab_processor = text.VocabularyProcessor(
        MAX_DOCUMENT_LENGTH, vocabulary=vocab)
    tokens = vocab_processor.fit_transform(texts)
    print list(tokens)


if __name__ == '__main__':
    train_data = ['John likes to watch movies. Mary likes movies too.',
            'John also likes to watch football games.']

    vectorize_using_tensor_flow(train_data)
