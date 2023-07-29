import random

rectangles = []
rectangle_average = []

for i in range(0,1000):
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
    return output_neuron




def train(rectangles, hidden_layer):
    outputs = []
    for rectangle in rectangles:
        output = model(rectangle, hidden_layer)
        outputs.append(output)

    mean_squared_error(outputs, rectangle_average)

    for index, _ in enumerate(hidden_layer):
        learning_rate = 0.1
        hidden_layer[index] -= learning_rate * hidden_layer[index].slope


    return outputs




hidden_layer = [0.98, 0.4, 0.86, -0.08]
train(rectangles, hidden_layer)



