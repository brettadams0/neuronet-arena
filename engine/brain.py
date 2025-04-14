# engine/brain.py
# 🧠 Simple feedforward neural network for agent decision-making

import random
import math
import copy

class Brain:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Xavier Initialization
        self.w1 = [[random.uniform(-1, 1) / math.sqrt(input_size) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [[random.uniform(-1, 1) / math.sqrt(hidden_size) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size

    def forward(self, inputs):
        h = [0.0] * self.hidden_size
        o = [0.0] * self.output_size

        # Input to Hidden
        for j in range(self.hidden_size):
            h[j] = sum(inputs[i] * self.w1[i][j] for i in range(self.input_size)) + self.b1[j]
            h[j] = math.tanh(h[j])

        # Hidden to Output
        for k in range(self.output_size):
            o[k] = sum(h[j] * self.w2[j][k] for j in range(self.hidden_size)) + self.b2[k]
            o[k] = 1 / (1 + math.exp(-o[k]))  # sigmoid activation

        return o

    def clone(self):
        return copy.deepcopy(self)

    def mutate(self, rate=0.1, strength=0.5):
        def mutate_value(val):
            return val + random.uniform(-strength, strength) if random.random() < rate else val

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.w1[i][j] = mutate_value(self.w1[i][j])
        for j in range(self.hidden_size):
            self.b1[j] = mutate_value(self.b1[j])

        for j in range(self.hidden_size):
            for k in range(self.output_size):
                self.w2[j][k] = mutate_value(self.w2[j][k])
        for k in range(self.output_size):
            self.b2[k] = mutate_value(self.b2[k])