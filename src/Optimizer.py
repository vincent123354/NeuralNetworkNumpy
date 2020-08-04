class GradientDescent:
    def __init__(self, params, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.params = params

    def step(self):
        for param in self.params:
            param[0].values -= self.learning_rate * param[0].grad
            param[1].values -= self.learning_rate * param[1].grad