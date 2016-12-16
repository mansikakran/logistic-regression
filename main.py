# Sources
# http://homel.vsb.cz/~pla06/files/mad3/MAD3_03.pdf
# http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect09.pdf
# http://sebastianruder.com/optimizing-gradient-descent/index.html#stochasticgradientdescent
# http://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning

import traceback
import random
import time
from math import exp


class LR_SGD:
    """
    Logistic regression using stochastic gradient descend
    """
    coefficients = []
    dataset = []
    accuracy = 0

    def __init__(self, dataset, normalized=False, coef=None):
        # normalize if not
        if not normalized:
            self.dataset = self.normalize_data_set(dataset)
        else:
            self.dataset = self.dataset

        # init coefficients
        if coef is None:
            self.coefficients = [0] * len(dataset[0])
        else:
            self.coefficients = coef

    def print_coefficients(self):
        print("Coefficients: ", self.coefficients)

    def normalize_data_set(self, dataset=None):
        if dataset is None:
            dataset = self.dataset

        min_max = []
        for i in range(len(dataset[0])):
            col_values = [row[i] for row in dataset]
            min_max.append([min(col_values), max(col_values)])

        for x in dataset:
            for i in range(len(x)):
                x[i] = (x[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])
        return dataset

    def likelihood(self, item, coef=None):
        if coef is None:
            coef = self.coefficients

        # 1 + e ^ -(b0 + bi*xi)
        # b0 is the first item
        cl = coef[0]
        for i in range(len(item) - 1):
            cl += coef[i + 1] * item[i]
        return 1.0 / (1.0 + exp(-cl))

    def analyze_dataset(self, verbose=True):
        count_ok = 0
        for x in self.dataset:
            r = self.likelihood(x)
            # print("E: {}, P: {:0.3f}, R: {}".format(x[-1], r, round(r)))
            if x[-1] == round(r):
                count_ok += 1
        # Accuracy
        self.accuracy = count_ok / len(self.dataset)
        if verbose:
            print("Correct predictions: {}, incorrect: {}, "
                  "accuracy: {:0.3f}%".format(count_ok,
                                              len(self.dataset) - count_ok,
                                              self.accuracy*100))

    def calculate_coefficients(self, alpha, n, shuffle=False, verbose=False):
        """
        Calculate the coefficients for the datatset in this class
        :param alpha: step size - should find using binary search
        :param n: number of repeats (epochs)
        :param shuffle: if we should shuffle dataset on each epoch
        :param verbose:
        :return:
        """
        for c in range(n):
            # Shuffle the dataset at each epoch for better results
            # http://sebastianruder.com/optimizing-gradient-descent/index.html#shufflingandcurriculumlearning
            if shuffle:
                random.shuffle(self.dataset)

            total_error = 0
            for item in self.dataset:
                cl = self.likelihood(item)
                er = item[-1] - cl
                self.coefficients[0] += alpha * er * cl * (1.0 - cl)
                # update all other
                for z in range(len(item) - 1):
                    self.coefficients[z + 1] += alpha * er * cl * (1.0 - cl) * \
                                                item[z]
                total_error += er * er  # ^2 cos negatives
            if verbose:
                print("Iteration: {}, error: {}".format(c, total_error))
                # self.print_coefficients()


def load_dataset_from_file(file_path):
    dataset = []
    try:
        f = open(file_path)
        lines = f.read().splitlines()
        for x in lines:
            dataset.append(list(map(float, x.split(','))))
        f.close()
    except Exception as e:
        print(traceback.format_exc())
        dataset = None
    finally:
        return dataset


# Split dataset 66%, 66 used for calc / 44 for validation
def split_dataset(dataset, percent=0.66, shuffle=True, verbose=True):
    if verbose:
        print("Splitting the dataset by {}%".format(percent*100))
    if shuffle:
        random.shuffle(dataset)
    s = round(len(dataset) * percent)
    return [dataset[:s], dataset[s:]]


def run_simulation(filename, random =False, verbose=False):
    s = time.clock()
    d = load_dataset_from_file(filename)
    if verbose:
        print("Dataset {} loaded. Number of Instances: {}"
              " Loading dataset took: {:0.4f}s"
              .format(filename, len(d), time.clock() - s))

    dat = split_dataset(d, 0.66, random, verbose)

    t = time.clock()
    s = time.clock()
    train = LR_SGD(dat[0])
    train.calculate_coefficients(0.1, 300, True)
    if verbose:
        print("Creating the model took: {:0.4f}s".format(time.clock() - s))

        train.print_coefficients()

    s = time.clock()
    cof = train.coefficients
    test = LR_SGD(dat[1], False, cof)
    test.analyze_dataset()
    if verbose:
        print("Analyzing dataset took: {:0.4f}s".format(time.clock() - s))
        print("Total time to analyze: {:0.4f}s\n".format(time.clock() - t))
    return test.accuracy


def run_multiple_times(file, r, v):
    f = []
    for x in range(r):
        f.append(run_simulation(file, True, v))
    tsum = 0
    for x in f:
        tsum += x
    print("Total accuracy in {} runs: {:0.3f}%\n".format(r, (tsum / r) * 100))

if __name__ == "__main__":
    run_simulation('data/pima-indians-diabetes.csv', False, True)
    run_multiple_times('data/pima-indians-diabetes.csv', 10, False)
