class Model:

    def __init__(self):
        pass

    def predict(self, input):
        raise NotImplementedError('Must implement this method in sub class.')

    def fit(self):
        raise NotImplementedError('Must implement this method in sub class.')
