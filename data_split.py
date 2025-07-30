import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split X and y into train/test
    """

    # basic check 
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    # set seed if user gave random_state
    if random_state is not None:
        np.random.seed(random_state)

    # make an array of indices like [0,1,2,...]
    indices = np.arange(X.shape[0])

    # shuffle them randomly
    np.random.shuffle(indices)

    # how many go to test
    test_count = int(len(X) * test_size)

    # first part is test, rest is train
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]

    # now split X and y based on these
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
