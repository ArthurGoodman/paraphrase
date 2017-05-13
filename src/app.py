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


def _features(a, b):
    features = {}

    features[0] = 0

    return features


def _build_features(data):
    features = []

    for case in data:
        feature_vector = _features(case[0], case[1])
        label = case[2]

        features.append((feature_vector, label))

    return features


def _classify(a, b):
    return classifier.classify(_features(a, b))


def _train():
    train_data = _read_data('../data/msr_paraphrase_train.txt')
    train_features = _build_features(train_data)

    # print(train_features)

    print('Training...')

    classifier.train(train_features)


def _test():
    test_data = _read_data('../data/msr_paraphrase_test.txt')

    print('Testing...\n')

    tp, tn, fp, fn = 0, 0, 0, 0

    for case in test_data:
        result = _classify(case[0], case[1])
        label = case[2]

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

    print('Accuracy: %i%%' % int(accuracy * 100))
    print('F1: %f' % f1)


def main():
    """Main."""
    _train()
    _test()


main()
