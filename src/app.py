"""Paraprase identification."""


def _tokenize(doc, filter_stopwords=True, normalize='lemma'):
    import nltk.corpus
    from nltk.stem import PorterStemmer, WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, wordpunct_tokenize
    from string import punctuation

    # use NLTK's default set of english stop words
    stops_list = nltk.corpus.stopwords.words('english')

    if normalize == 'lemma':
        # lemmatize with WordNet
        normalizer = WordNetLemmatizer()
    elif normalize == 'stem':
        # stem with Porter
        normalizer = PorterStemmer()

    # tokenize the document into sentences with NLTK default
    sents = sent_tokenize(doc)
    # tokenize each sentence into words with NLTK default
    tokenized_sents = [wordpunct_tokenize(sent) for sent in sents]
    # filter out "bad" words, normalize good ones
    normalized_sents = []
    for tokenized_sent in tokenized_sents:
        good_words = [word for word in tokenized_sent
                      # filter out too-long words
                      if len(word) < 25
                      # filter out bare punctuation
                      if word not in list(punctuation)]
        if filter_stopwords is True:
            good_words = [word for word in good_words
                          # filter out stop words
                          if word not in stops_list]
        if normalize == 'lemma':
            normalized_sents.append(
                [normalizer.lemmatize(word, 'v') for word in good_words])
        elif normalize == 'stem':
            normalized_sents.append([normalizer.stem(word)
                                     for word in good_words])
        else:
            normalized_sents.append([word for word in good_words])

    return normalized_sents


def _read_data(file_name):
    first_line = True

    data = []

    with open(file_name) as f:
        for line in f:
            if (first_line):
                first_line = False
                continue

            array = line.split('\t')
            data.append((array[3], array[4], int(array[0])))

    return data


def _features(a, b):
    import nltk

    a_words = [word[0] for word in a]
    b_words = [word[0] for word in b]

    features = {}

    bleu = nltk.translate.bleu_score.sentence_bleu(
        [a_words], b_words, weights=[1])

    features[len(features)] = bleu

    return features


def _preprocess(sentence):
    import nltk

    sentence = sentence.translate(dict.fromkeys(map(ord, ".!?"), None))

    tokens = _tokenize(sentence, False)[0]
    tokens = [token for token in tokens if token[0].isalnum()]

    return nltk.pos_tag(tokens)


def _build_features(data):
    features = []

    for case in data:
        feature_vector = _features(_preprocess(case[0]), _preprocess(case[1]))
        label = case[2]

        features.append((feature_vector, label))

    return features


def _train(classifier):
    print('Preprocessing training data...')

    train_data = _read_data('../data/msr_paraphrase_train.txt')
    train_features = _build_features(train_data)

    print('Training...')

    classifier.train(train_features)


def _test(classifier):
    print('Preprocessing test data...')

    test_data = _read_data('../data/msr_paraphrase_test.txt')
    test_features = _build_features(test_data)

    print('Testing...')

    tp, tn, fp, fn = 0, 0, 0, 0

    for case in test_features:
        result = classifier.classify(case[0])
        label = case[1]

        if result == label:
            if result == 1:
                tp += 1
            else:
                tn += 1
        else:
            if result == 1:
                fp += 1
            else:
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1 = 2 * precision * recall / (precision + recall)

    print('')
    print('Accuracy: %i%%' % int(accuracy * 100))
    print('F1: %f' % f1)


def main():
    """Main."""
    from sklearn.svm import SVC
    from nltk.classify import SklearnClassifier

    classifier = SklearnClassifier(SVC(kernel="rbf"), sparse=False)

    _train(classifier)
    _test(classifier)

main()
