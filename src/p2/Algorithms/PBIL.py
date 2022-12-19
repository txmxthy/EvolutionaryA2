import random


class ProbabilityBasedIncrementalLearning(object):
    def __init__(self, dim, learning_rate, mut_prob, mut_shift, pop_sizer):
        self.prob_vector = [0.5] * dim
        self.learning_rate = learning_rate
        self.mut_prob = mut_prob
        self.mut_shift = mut_shift
        self.pop_sizer = pop_sizer

    def sample(self):
        """
        Sample a new individual from the probability vector
        @return:
        """
        return (random.random() < prob for prob in self.prob_vector)

    def generate(self, pop_sizer):
        """
        Generate a new population
        @param pop_sizer:
        @return:
        """
        return [pop_sizer(self.sample()) for _ in range(self.pop_sizer)]

    def update(self, population):
        """
        Update the probability vector
        @param population:
        @return:
        """

        ideal = max(population, key=lambda ind: ind.fitness.values[0] - ind.fitness.values[1])
        for i, value in enumerate(ideal):
            # Update
            self.prob_vector[i] *= 1.0 - self.learning_rate
            self.prob_vector[i] += value * self.learning_rate

            # Mutate
            if random.random() < self.mut_prob:
                self.prob_vector[i] *= 1.0 - self.mut_shift
                self.prob_vector[i] += random.randint(0, 1) * self.mut_shift
