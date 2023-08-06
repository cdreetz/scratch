import numpy
import torch as torch
import random

rectangles = []
rectangle_average = []


for i in range(0, 1000):
    # Generate a 2x2 rectangle [0.1, 0.8, 0.6, 1.0]
    rectangle = [round(random.random(), 1),
                 round(random.random(), 1),
                 round(random.random(), 1),
                 round(random.random(), 1)]
    rectangles.append(rectangle)
    rectangle_average.append(sum(rectangle) / 4)


def mean_squared_error(actual, expected):
    error_sum = 0
    for a, b in zip(actual, expected):
        error_sum += (a - b) ** 2
    return error_sum / len(actual)




def model(rectangle, hidden_layer):
    output_neuron = 0.
    for index, input_neuron in enumerate(rectangle):
        output_neuron += input_neuron * hidden_layer[index]
    return max(0, output_neuron)


def train(rectangles, hidden_layer):
    outputs = []
    for rectangle in rectangles:
        output = model(rectangle, hidden_layer)
        outputs.append(output)


    error = mean_squared_error(outputs, rectangle_average)

    error.backward()

    for index, _ in enumerate(hidden_layer):
        learning_rate = 0.1
        hidden_layer.data[index] -= learning_rate * hidden_layer.grad.data[index]

    hidden_layer.grad.zero_()
    return error



hidden_layer = torch.tensor([0.98, 0.4, 0.86, -0.08], requires_grad=True)


#train(rectangles, hidden_layer)

#print(f"After: {model([0.2,0.5,0.4,0.7], hidden_layer)}")



for epoch in range(100):
    error = train(rectangles, hidden_layer)


print(model([0.2,0.5,0.4,0.7], hidden_layer).item())

