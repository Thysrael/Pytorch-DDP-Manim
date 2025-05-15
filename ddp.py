import numpy as np
import pandas as pd
from manim import *

HEADER_FONT_SIZE = 20
ANIMATION_RUN_TIME = 0.1

# Activation functions
def relu(X):
    return np.maximum(0,X)

def softmax(X):
    Z = X - max(X)
    numerator = np.exp(Z)
    denominator = np.sum(numerator)
    return numerator / denominator

# Calculates the output of a given layer
def calculate_layer_output(w, prev_layer_output, b, actv_func):
    # Steps 1 & 2
    g = w @ prev_layer_output + b

    # Step 3
    return actv_func(g)

# Initialize weights & biases
def init_layer_params(row, col):
    w = np.random.randn(row, col)
    b = np.random.randn(row, 1)
    return w, b

# Calculate ReLU derivative
def relu_derivative(g):
    # g shape: (n, 1)
    derivative = g.copy()
    derivative[derivative <= 0] = 0
    derivative[derivative > 0] = 1
    # derivative.T[0] shape: (n) is the correct shape for np.diag()
    return np.diag(derivative.T[0])

# Calculate Softmax derivative
def softmax_derivative(o):
    o = o.flatten()
    return np.diag(o) - np.outer(o, o)

def layer_backprop(previous_derivative, layer_output, previous_layer_output
                   , w, actv_func):
    # 1. Calculate the derivative of the activation func
    dh_dg = None
    if actv_func is relu:
        dh_dg = relu_derivative(layer_output)
    elif actv_func is softmax:
        dh_dg = softmax_derivative(layer_output)

    # 2. Apply chain rule to get derivative of Loss function with respect to:
    dL_dg = dh_dg @ previous_derivative # activation function

    # 3. Calculate the derivative of the linear function with respect to:
    dg_dw = previous_layer_output.T     # a) weight matrix
    dg_dh = w.T                         # b) previous layer output
    dg_db = 1.0                         # c) bias vector

    # 4. Apply chain rule to get derivative of Loss function with respect to:
    dL_dw = dL_dg @ dg_dw               # a) weight matrix
    dL_dh = dg_dh @ dL_dg               # b) previous layer output
    dL_db = dL_dg * dg_db               # c) bias vector

    return dL_dw, dL_dh, dL_db

def gradient_descent(w, b, dL_dw, dL_db, learning_rate):
    w -= learning_rate * dL_dw
    b -= learning_rate * dL_db
    return w, b

def get_prediction(o):
    return np.argmax(o)

# Compute Accuracy (%) across all training data
def compute_accuracy(train, label, w1, b1, w2, b2, w3, b3):
    # Set params
    correct = 0
    total = train.shape[0]

    # Iterate through training data
    for index in range(0, train.shape[0]):
        # Select a single data point (image)
        X = train[index: index + 1,:].T

        # Forward pass: compute Output/Prediction (o)
        h1 = calculate_layer_output(w1, X, b1, relu)
        h2 = calculate_layer_output(w2, h1, b2, relu)
        o = calculate_layer_output(w3, h2, b3, softmax)

        # If prediction matches label Increment correct count
        if label[index] == get_prediction(o):
            correct += 1

    # Return Accuracy (%)
    return (correct / total) * 100

class VisualNode:
    def __init__(self, init_image, w2, w3):

        self.w2 = w2
        self.w3 = w3

        self.group = VGroup()
        self.input_image = self.create_input_image(init_image)
        self.input_image.shift(UP)
        self.input_image_text = Text("Input Image", font_size=HEADER_FONT_SIZE)
        self.input_image_text.next_to(self.input_image, UP)

        num_nodes = 10
        self.h1_node_group, self.h1_nodes_list = self.create_nodes(num_nodes)
        self.h2_node_group, self.h2_nodes_list = self.create_nodes(num_nodes)
        self.o_node_group, self.o_nodes_list = self.create_nodes(num_nodes)

        self.h1_node_group.shift(2 * RIGHT)
        self.h2_node_group.shift(5 * RIGHT)
        self.o_node_group.shift(8 * RIGHT)

        self.connections_1 = self.create_connections(self.h1_nodes_list, self.h2_nodes_list, self.w2)
        self.connections_2 = self.create_connections(self.h2_nodes_list, self.o_nodes_list, self.w3)

        self.prediction = self.create_prediction_text("...")
        self.prediction.next_to(self.input_image, DOWN)
        self.prediction.shift(DOWN)

        self.gradient = self.create_gradient(np.zeros((10, 10)), np.zeros((10, 10)))
        self.gradient.next_to(self.h2_node_group, DOWN)
        self.gradient_text = Text("Gradient", font_size=HEADER_FONT_SIZE)
        self.gradient_text.next_to(self.gradient, LEFT)

        # the order here determines the play order
        self.group.add(self.input_image_text)
        self.group.add(self.input_image)
        self.group.add(self.h1_node_group)
        self.group.add(self.h2_node_group)
        self.group.add(self.o_node_group)
        self.group.add(self.connections_1)
        self.group.add(self.connections_2)
        self.group.add(self.prediction)
        self.group.add(self.gradient_text)
        self.group.add(self.gradient)


    def create_input_image(self, training_image):
        # Initialise params
        square_count = training_image.shape[0]
        rows = np.sqrt(square_count)

        # Create list of squares to represent pixels
        squares = [
            Square(fill_color=WHITE,
                   fill_opacity=training_image[i],
                   stroke_width=0.5).scale(0.03)
            for i in range(square_count)
        ]

        # Place all the squares into a VGroup and arrange into a 28x28 grid
        group = VGroup(*squares).arrange_in_grid(rows=int(rows), buff=0)

        return group

    def animate_input_image(self, new_data):
        new_image = self.create_input_image(new_data)
        new_image.move_to(self.input_image.get_center())
        old_image = self.input_image
        self.input_image = new_image
        return Transform(old_image, new_image, run_time=ANIMATION_RUN_TIME)

    def create_nodes(self, num_nodes, layer_output=None):
        # Create VGroup & list to hold created nodes
        node_group = VGroup()
        nodes = []

        # Create list of circles to represent nodes
        for i in range(num_nodes):
            # Set fill opacity to 0
            opacity = 0.0
            text = "0.0"
            # If a layer output has been passed and the max value is not 0
            if layer_output is not None and np.max(layer_output) != 0.0:
                # Set opacity as normalised layer output value
                opacity = (layer_output[i] / np.max(np.absolute(layer_output)))[0]
                # Set text as layer output
                text = f'{layer_output[i][0]:.1f}'

            # Create node
            node = Circle(radius=0.23,
                          stroke_color=WHITE,
                          stroke_width=1.0,
                          fill_color=GRAY,
                          fill_opacity=opacity)

            # Add to nodes list
            nodes += [node]

            fill_text = Text(text, font_size=12)
            # Position fill text in circle
            fill_text.move_to(node)

            # Group fill text and node and add to node_group
            group = VGroup(node, fill_text)
            node_group.add(group)


        # Arrange & position node_group
        node_group.arrange(DOWN, buff=0.2)

        return node_group, nodes

    def create_connections(self, left_layer_nodes, right_layer_nodes, w):
        # Create VGroup to hold created connections
        connection_group = VGroup()

        # Iterate through right layer nodes
        for l in range(len(right_layer_nodes)):
            # Iterate through left layer nodes
            for r in range(len(left_layer_nodes)):
                # Calculate opacity from normalised weight matrix values
                opacity = 0.0 if np.max(np.absolute(w[l, :])) == 0.0 \
                    else w[l, r] / np.max(np.absolute(w[l, :]))
                # Set colour
                colour = TEAL_E if opacity >= 0 else GREEN_E

                # Create connection line
                line = Line(start=right_layer_nodes[l].get_edge_center(LEFT),
                            end=left_layer_nodes[r].get_edge_center(RIGHT),
                            color=colour,
                            stroke_opacity=abs(opacity))

                # Add to connection group
                connection_group.add(line)

        return connection_group

    def create_prediction_text(self, prediction):
        # Create group
        prediction_text_group = VGroup()

        # Create & position text
        prediction_text = Text(f'{prediction}', font_size=2 * HEADER_FONT_SIZE, color=BLUE)

        # Create text box (helps with positioning Prediction Header)
        prediction_text_box = Square(fill_opacity=0,
                                     stroke_opacity=0,
                                     side_length=0.75)
        prediction_text_box.move_to(prediction_text)

        # Create Header Text
        prediction_header = Text("Prediction", font_size=1.5 * HEADER_FONT_SIZE)
        prediction_header.next_to(prediction_text_box, UP)

        # Group items
        prediction_text_group.add(prediction_header)
        prediction_text_group.add(prediction_text)
        prediction_text_group.add(prediction_text_box)

        return prediction_text_group

    def normalize(vector):
        min_val = np.min(vector)
        max_val = np.max(vector)
        if max_val == min_val:
            return np.zeros_like(vector)
        return (vector - min_val) / (max_val - min_val)

    def create_gradient(self, gradient1, gradient2):
        # gradient: (10, 10) only the hidden layer weight gradient
        # we have two hidden layers
        rows = 4
        scale = 0
        gf1 = normalize(gradient1.flatten())
        gf2 = normalize(gradient2.flatten())

        squares = [
            Square(fill_color=TEAL_A,
                   fill_opacity=gf1[i] + scale,
                   stroke_width=0.6,
                   stroke_color=TEAL_C).scale(0.08)
            for i in range(gf1.shape[0])
        ]

        squares += [
            Square(fill_color=TEAL_B,
                   fill_opacity=gf2[i] + scale,
                   stroke_width=0.6,
                   stroke_color=TEAL_D).scale(0.08)
            for i in range(gf2.shape[0])
        ]

        group = VGroup(*squares).arrange_in_grid(rows=int(rows), buff=0)

        return group

class VisualiseNeuralNetwork(Scene):
    def construct(self):
        ### INITIALISE NEURAL NET PARAMETERS ###
        # Extract MNIST csv data into train & test variables
        train = np.array(pd.read_csv('./digit-recognizer/train.csv', delimiter=','))
        test = np.array(pd.read_csv('./digit-recognizer/test.csv', delimiter=','))

        # Extract the first column of the training dataset into a label array
        label = train[:, 0]
        # The train dataset now becomes all columns except the first
        train = train[:, 1:]
        np.random.seed(42)

        # Initialise vector of all zeroes with 10 columns and the same number
        # of rows as the label array
        Y = np.zeros((label.shape[0], 10))

        # assign a value of 1 to each column index matching the label value
        Y[np.arange(0, label.shape[0]), label] = 1.0

        # Normalize test & training dataset
        train = train / 255
        test = test / 255

        # Set hyperparameter(s)
        learning_rate = 0.01
        epoch = 20
        previous_accuracy = 100
        accuracy = 0

        # Randomly initialize weights & biases
        w1, b1 = init_layer_params(10, 784)  # Hidden Layer 1
        w2, b2 = init_layer_params(10, 10)  # Hidden Layer 2
        w3, b3 = init_layer_params(10, 10)  # Output Layer

        init_image_id_1 = np.random.randint(low=0, high=train.shape[0])
        init_image_1 = train[init_image_id_1:init_image_id_1 + 1, :].T

        init_image_id_2 = np.random.randint(low=0, high=train.shape[0])
        init_image_2 = train[init_image_id_2:init_image_id_2 + 1, :].T

        node_1 = VisualNode(init_image_1, w2, w3)
        node_2 = VisualNode(init_image_2, w2, w3)

        group_1 = node_1.group
        group_2 = node_2.group
        group_1.next_to(group_2, LEFT)
        group_2.shift(2 * RIGHT)

        status = Text(f'Epoch: {0}\tAccuracy: {0:05.2f}%', font_size=1.8 * HEADER_FONT_SIZE)
        status.shift(4 * UP)
        self.add(status)

        self.play(
            Create(group_1),
            Create(group_2)
        )
        self.wait(0.5)

        # self.play(node_1.animate_input_image(init_image_2))
