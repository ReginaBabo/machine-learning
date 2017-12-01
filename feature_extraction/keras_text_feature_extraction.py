import keras
from keras.preprocessing import text

def vectorize_using_keras(texts, val_texts):
    # word tokens
    num_words = 25
    tokenizer = text.Tokenizer(num_words=num_words) 
    # Skipped the char_level setting
    tokenizer.fit_on_texts(texts)

    print 'Document count'
    print tokenizer.document_count

    print 'Dictionary mapping words to the number of times they appeared on during fit'
    print tokenizer.word_counts

    print 'Dictionary mapping words to their rank/index'
    # 0th index is reserved for missing tokens
    # Models like CNN use this for reserved tokens?
    # Models like RNN just ignore these?
    print tokenizer.word_index

    print 'Dictionary mapping words to the number of documents/texts they appeared on during fit'
    print tokenizer.word_docs

    print texts[0]
    print 'Sequence corresponding to the texts'
    # This is just the sequence of indices corresponding to the given text
    x_train_sequences = tokenizer.texts_to_sequences(texts)
    print x_train_sequences[0]

    print 'Binary matrix corresponding to the texts'
    # n-grams not available here. It just converts all filter char to space
    # and splits on space.
    x_train_matrix = tokenizer.texts_to_matrix(texts)
    print x_train_matrix[0]

    print 'Count matrix corresponding to the texts'
    x_train_matrix = tokenizer.texts_to_matrix(texts, mode='count')
    print x_train_matrix[0]

    print 'Tfidf matrix corresponding to the texts'
    x_train_matrix = tokenizer.texts_to_matrix(texts, mode='tfidf')
    print x_train_matrix[0]

    print 'Freq matrix corresponding to the texts'
    # This is the count divided by total number of words?
    x_train_matrix = tokenizer.texts_to_matrix(texts, mode='freq')
    print x_train_matrix[0]

    print 'One hot'
    # Same as hashing trick. Wrapper to it for default hash function.
    one_hot_response = text.one_hot(texts[0], num_words)
    print one_hot_response

    print 'Text to word sequence'
    word_sequence = text.text_to_word_sequence(texts[0])
    print word_sequence

    print 'Hashing trick'
    # Defaults to python hash function which can be md5 or any function
    hashing_trick_response = text.hashing_trick(texts[0], num_words)
    print hashing_trick_response

if __name__ == '__main__':
    train_data = ['John likes to watch movies. Mary likes movies too.',
            'John also likes to watch football games.']
    test_data = ['John and Mary both like movies.']

    vectorize_using_keras(train_data, test_data)
