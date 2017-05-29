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


def _match_ngram(a, b, match):
    if len(a) != len(b):
        return False

    for i in range(len(a)):
        if not match(a[i], b[i]):
            return False

    return True


def _count(gram, rf, n, match):
    from nltk import ngrams

    count = 0

    for rf_gram in ngrams(rf, n):
        if _match_ngram(gram, rf_gram, match):
            count += 1

    return count


def _bleu(rf, tr, n, match):
    from nltk import ngrams
    import math

    bleu = 0

    for i in range(1, n + 1):
        counts = {}
        ln = 0

        for tr_gram in ngrams(tr, i):
            if tr_gram not in counts:
                counts[tr_gram] = 0

            ln += 1

            if _count(tr_gram, rf, i, match) > 0:
                counts[tr_gram] += 1

        max_counts = {}

        for gram in counts:
            max_counts[gram] = _count(gram, rf, i, match)

        mn = 0

        for gram in counts:
            mn += min(max_counts[gram], counts[gram])

        v = mn / ln

        if v > 0:
            bleu += 1 / n * math.log(v)

    return min(1, math.exp(1 - len(rf) / len(tr))) * \
        math.pow(math.exp(bleu), 1 / n)


def _literal_match(a, b):
    return a[0] == b[0]


synonyms = {}


def _synonym_match(a, b):
    if a[0] == b[0]:
        return True

    from itertools import chain
    from nltk.corpus import wordnet as wn

    if b[0] not in synonyms:
        synsets = wn.synsets(b[0])
        lemmas = set(chain.from_iterable(
            [word.lemma_names() for word in synsets]))

        synonyms[b[0]] = lemmas

    return a[0] in synonyms[b[0]]


def _pos_match(a, b):
    return a[1] == b[1]

synsets = {}


def _synsets(word):
    from nltk.corpus import wordnet as wn

    if word in synsets:
        return synsets[word]

    s = wn.synsets(word)

    synsets[word] = s
    return s

distances = {}


def _dist(w1, w2):
    sets1 = _synsets(w1[0])
    sets2 = _synsets(w2[0])

    min_sim = 1e10
    found_at_least_one = False

    for set1 in sets1:
        for set2 in sets2:
            sim = set1.wup_similarity(set2)

            if sim is not None:
                min_sim = min(min_sim, sim)
                found_at_least_one = True

    if found_at_least_one:
        return min_sim
    else:
        return None


def _dist_cached(w1, w2):
    t = '%s,%s' % (w1, w2)

    if t in distances:
        return distances[t]

    d = _dist(w1, w2)

    distances[t] = d
    return d


def _new_super_effective_feature_by_dima(a, b):
    sim_sum = 0
    sim_count = 0

    for w1 in a:
        min_sim = 1e10
        found_at_least_one = False

        for w2 in b:
            d = _dist_cached(w1[0], w2[0])

            if d is not None:
                min_sim = min(min_sim, d)
                found_at_least_one = True

        if found_at_least_one:
            sim_sum += min_sim
            sim_count += 1

    return sim_sum / sim_count


def _features(a, b):
    features = {}

    features['bleu1'] = _bleu(a, b, 1, _literal_match)
    # features['bleu2'] = _bleu(a, b, 2, _literal_match)
    # features['bleu3'] = _bleu(a, b, 3, _literal_match)
    # features['bleu4'] = _bleu(a, b, 4, _literal_match)

    # features['pos_bleu1'] = _bleu(a, b, 1, _pos_match)
    # features['pos_bleu2'] = _bleu(a, b, 2, _pos_match)
    # features['pos_bleu3'] = _bleu(a, b, 3, _pos_match)
    # features['pos_bleu4'] = _bleu(a, b, 4, _pos_match)

    # features['syn_bleu1'] = _bleu(a, b, 1, _synonym_match)
    # features['syn_bleu2'] = _bleu(a, b, 2, _synonym_match)
    # features['syn_bleu3'] = _bleu(a, b, 3, _synonym_match)
    # features['syn_bleu4'] = _bleu(a, b, 4, _synonym_match)

    features[
        'new_super_effective'
        '_feature_by_dima'] = _new_super_effective_feature_by_dima(a, b)

    return features


def _preprocess(sentence):
    import nltk

    sentence = sentence.translate(dict.fromkeys(map(ord, ".!?"), None))

    tokens = _tokenize(sentence, False)[0]
    tokens = [token.lower() for token in tokens if token[0].isalnum()]

    return nltk.pos_tag(tokens)


def _progress_bar(progress, length):
    return '[' + '=' * int(progress * length) + \
        ' ' * int((1 - progress) * length) + ']'


def _build_features(data):
    import sys

    features = []

    i = 0

    for case in data:
        feature_vector = _features(_preprocess(case[0]), _preprocess(case[1]))
        label = case[2]

        sys.stdout.write('\r%s %i%% ' %
                         (_progress_bar(i / len(data), 70),
                          int(100 * (i + 1) / len(data))))

        sys.stdout.flush()

        i += 1

        features.append((feature_vector, label))

    print('')

    # print(features)

    with open('feature_names.csv', 'wt') as file:
        feature_vector = features[0]

        s = ''

        for feature_name in feature_vector[0]:
            if len(s) > 0:
                s += ','

            s += feature_name

        s += '\n'

        file.write(s)

    with open('features.csv', 'wt') as file:
        for feature_vector in features:
            s = ''

            for feature_name in feature_vector[0]:
                if len(s) > 0:
                    s += ','

                s += str(feature_vector[0][feature_name])

            s += ',%i\n' % feature_vector[1]

            file.write(s)

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
    print('True positive: %i' % tp)
    print('True negative: %i' % tn)
    print('False positive: %i' % fp)
    print('False negative: %i' % fn)

    print('')
    print('Accuracy: %i%%' % int(accuracy * 100))
    print('Precision: %i%%' % int(precision * 100))
    print('Recall: %i%%' % int(recall * 100))
    print('F1: %i%%' % int(f1 * 100))


def main():
    """Main."""
    from sklearn.svm import SVC
    from nltk.classify import SklearnClassifier

    classifier = SklearnClassifier(SVC(kernel="rbf"), sparse=False)

    _train(classifier)
    _test(classifier)

main()
