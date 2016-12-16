# MAD III- Project

## Logistic regression using stochastic gradient descent - implemented in Python


### Experiments
**Dataset:** pima-indians-diabetes with 768 instances

**Validation used**: Percentage split - in the tests used as 66%

**Example output**:
```
Dataset pima-indians-diabetes.csv loaded. Number of Instances: 768 Loading dataset took: 0.0089s
Splitting the dataset by 66.0%
Creating the model took: 1.7120s
Coefficients:  [-7.582051426755043, 1.854301929346409, 7.193768376521267, -1.70601044011557, -0.5132182256460746, 0.3630667567006922, 4.938894187944461, 3.0229812111891645, 0.7150993199638532]
Correct predictions: 201, incorrect: 60, accuracy: 77.011%
Analyzing dataset took: 0.0034s
Total time to analyze: 1.7155s
```

**Compare to Weka:**

|         | Weka 3.6.13          | This implementation  |
| :-----:| :-----:| :-----:|
| Speed      | 0.01s | ~1.7s |
| Accuracy | 81.226%     |    80.843% |
| 10 runs accuracy | 78.651% | 76.782% |

**Notes**:
- The first accuracy is measured without the random shuffle
- When run the test 10 times dataset is always shuffled before split
- In a long run it seems that Weka has better results
 
 
 **Example output for 10 runs:**
```
Correct predictions: 196, incorrect: 65, accuracy: 75.096%
Correct predictions: 200, incorrect: 61, accuracy: 76.628%
Correct predictions: 201, incorrect: 60, accuracy: 77.011%
Correct predictions: 206, incorrect: 55, accuracy: 78.927%
Correct predictions: 203, incorrect: 58, accuracy: 77.778%
Correct predictions: 197, incorrect: 64, accuracy: 75.479%
Correct predictions: 201, incorrect: 60, accuracy: 77.011%
Correct predictions: 199, incorrect: 62, accuracy: 76.245%
Correct predictions: 200, incorrect: 61, accuracy: 76.628%
Correct predictions: 201, incorrect: 60, accuracy: 77.011%
Total accuracy in 10 runs: 76.782%
```

### Requirements to run 
1. python 3

### Sources
- https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/
- http://homel.vsb.cz/~pla06/files/mad3/MAD3_03.pdf
- http://www.cs.rpi.edu/~magdon/courses/LFD-Slides/SlidesLect09.pdf
- http://sebastianruder.com/optimizing-gradient-descent/index.html#stochasticgradientdescent
- http://stats.stackexchange.com/questions/189652/is-it-a-good-practice-to-always-scale-normalize-data-for-machine-learning

##### Performance measured on:
- i5-2410M CPU @ 2.30GHz
- Python 3.5.0