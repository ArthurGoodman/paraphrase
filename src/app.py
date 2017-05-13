"""Paraprase identification."""

from sklearn.svm import SVC
from nltk.classify import SklearnClassifier

classifier = SklearnClassifier(SVC(kernel="rbf"), sparse=False)


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


def _build_features(a, b):
    features = {}

    features[0] = 0

    return features


def _classify(a, b):
    return classifier.classify(_build_features('a', 'b'))


def main():
    """Main."""
    train_sentences = _read_data('../data/msr_paraphrase_train.txt')
    train_features = []

    for sentence in train_sentences:
        features = _build_features(sentence[0], sentence[1])
        label = sentence[2]

        train_features.append((features, label))

    print(train_features)

    print('Training...')

    classifier.train(train_features)

    print('Success!')

    a = 'Her life spanned years of incredible change for women.'
    b = 'Mary lived through an era of liberating reform for women.'

    print(_classify(a, b))


main()
