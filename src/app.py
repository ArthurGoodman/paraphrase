"""Paraprase identification."""

# from sklearn.svm import SVC
# from nltk.classify import SklearnClassifier

# train_features = [({'x': -1, 'y': -1}, 'one'),
#                   ({'x': -2, 'y': -1}, 'one'),
#                   ({'x': 1, 'y': 1}, 'two'),
#                   ({'x': 2, 'y': 1}, 'two')]

# classifier = SklearnClassifier(SVC(), sparse=False).train(train_features)

# print(classifier.classify({'x': 2, 'y': 0}))


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


def main():
    """Main."""
    train_sentences = _read_data('../data/msr_paraphrase_train.txt')
    print(train_sentences)
