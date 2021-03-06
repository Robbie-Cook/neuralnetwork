# import numpy as np
import math
import random
import copy

"""
    Initialisation
"""

learningConstant = 0.1
momentumConstant = 0.9
numberOfHiddenNodes = 3

inputPatterns = [
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0]
]

teacher = [
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
]
#
# inputPatterns = [
#     [0, 0],
#     [0, 1],
#     [1, 0],
#     [1, 1]
# ]
#
# teacher = [
#     [0.1],
#     [0.9],
#     [0.9],
#     [0.1]
# ]


nodes = {"input": [], "hidden": [], "output": []}

connections = {
    "firstLayer": [],
    "secondLayer": []
}

bias = {
    "hidden": [],
    "output": []
}

deltaArray = {
    "output": []
}

weightChangeArray = []

def initalise(input=[], teach=[], hiddenNodes=0):
    global learningConstant, momentumConstant, numberOfHiddenNodes, inputPatterns, teacher, nodes, connections, \
    deltaArray, weightChangeArray, learningConstant, momentumConstant, numberOfHiddenNodes

    if len(input) > 0:
        inputPatterns = input

    if len(teach) > 0:
        teacher = teach

    if hiddenNodes > 0:
        numberOfHiddenNodes = hiddenNodes

    numberOfInputNodes = len(inputPatterns[0])
    numberOfOutputNodes = len(teacher[0])

    # Delete all data
    for key in nodes.keys():
        nodes[key] = []

    for key in connections.keys():
        connections[key] = []

    for key in bias.keys():
        bias[key] = []

    for key in deltaArray.keys():
        deltaArray[key] = []

    weightChangeArray = []



    for i in range(0, numberOfInputNodes):
        nodes["input"].append(0)
    for i in range(0, numberOfHiddenNodes):
        nodes["hidden"].append(0)
    for i in range(0, numberOfOutputNodes):
        nodes["output"].append(0)


    for i in range(0, numberOfInputNodes):
        connections["firstLayer"].append([])
        for j in range(0, numberOfHiddenNodes):
            connections["firstLayer"][i].append(random.uniform(-1.5, 1.5))

    for i in range(0, numberOfHiddenNodes):
        connections["secondLayer"].append([])
        for j in range(0, numberOfOutputNodes):
            connections["secondLayer"][i].append(random.uniform(-1.5, 1.5))

    weightChangeArray = copy.deepcopy(connections)
    for key in weightChangeArray.keys():
        for i in range(0, len(weightChangeArray[key])):
            for j in range(0, len(weightChangeArray[key][i])):
                weightChangeArray[key][i][j] = 0

    keys = ["hidden", "output"]
    for key in keys:
        for i in range(0, len(nodes[key])):
            bias[key].append(random.uniform(-1.0, 1.0))

    for i in range(0, numberOfOutputNodes):
        deltaArray["output"].append(0)

def runNetwork(input, expectedOutput, teach=True, debug=False):
    global learningConstant, momentumConstant, numberOfHiddenNodes, inputPatterns, teacher, nodes, connections, \
    deltaArray, weightChangeArray, learningConstant, momentumConstant, numberOfHiddenNodes

    nodes["input"] = input

    """
        Calculate output
    """
    if debug:
        print("Initial connections:", connections)
        print("Bias", bias)
    # Calculate and store the activations for the output and hidden layer
    for layer in ["hidden", "output"]:
        # find the previous layer
        previousLayer = ""
        connectionsLayer = ""
        if layer == "output":
            previousLayer = "hidden"
            connectionsLayer = "secondLayer"
        else: # layer is hidden
            previousLayer = "input"
            connectionsLayer = "firstLayer"

        # Calculate the activation values for each node in the hidden or output layer
        for nodeIndex in range(0, len(nodes[layer])):
            netInput = 0
            for inputIndex in range(0, len(nodes[previousLayer])):
                netInput += connections[connectionsLayer][inputIndex][nodeIndex] * nodes[previousLayer][inputIndex]
            netInput += bias[layer][nodeIndex]
            netInput = round(netInput, 10)  # round net input
            activation = 1/(1+(1/math.exp(netInput)))
            activation = round(activation, 10)  # round activation
            nodes[layer][nodeIndex] = activation

    # Calculate population error
    sum = 0
    for i in range(0,len(nodes["output"])):
        sum += ((expectedOutput[i] - nodes["output"][i])**2)
    patternError = (0.5)*sum
    if debug:
        print("Pattern error: ", patternError)

    # Finish if not teaching
    if not teach:
        return patternError
    """
    Backprop
    """
    # Second layer
    for outputIndex in range(0, len(nodes["output"])): # for every output node
        output = nodes["output"][outputIndex]
        delta = (expectedOutput[outputIndex] - output) * (output) * (1-output)
        deltaArray["output"][outputIndex] = delta
        biasWeightChange = learningConstant * delta * 1
        bias["output"][outputIndex] += biasWeightChange
        for hiddenIndex in range(0, len(nodes["hidden"])):
            weightChange = learningConstant * delta * nodes["hidden"][hiddenIndex]
            weightChangeArray["secondLayer"][hiddenIndex][outputIndex] += weightChange

    # First Layer
    for hiddenIndex in range(0, len(nodes["hidden"])):
        output = nodes["hidden"][hiddenIndex]
        delta = output * (1 - output)
        sum = 0
        for outputIndex in range(0, len(nodes["output"])):
            sum += (deltaArray["output"][outputIndex] * connections["secondLayer"][hiddenIndex][outputIndex])
        delta *= sum
        biasWeightChange = learningConstant * delta * 1
        bias["hidden"][hiddenIndex] += biasWeightChange
        for inputIndex in range(0, len(nodes["input"])):
            weightChange = learningConstant * delta * nodes["input"][inputIndex]
            weightChangeArray["firstLayer"][inputIndex][hiddenIndex] += weightChange

    # Apply weight changes
    for hiddenIndex in range(0, len(nodes["hidden"])):
        for outputIndex in range(0, len(nodes["output"])):
            connections["secondLayer"][hiddenIndex][outputIndex] += weightChangeArray["secondLayer"][hiddenIndex][outputIndex]

    for inputIndex in range(0, len(nodes["input"])):
        for hiddenIndex in range(0, len(nodes["hidden"])):
            connections["firstLayer"][inputIndex][hiddenIndex] += weightChangeArray["firstLayer"][inputIndex][hiddenIndex]

    if debug:
        print("Weight changes", weightChangeArray)
        print("New weights", connections)
        print("Activations", nodes)
        print("Bias", bias)

    return patternError

"""
Run epoch
"""

def runEpoch(teach=True):
    global learningConstant, momentumConstant, numberOfHiddenNodes, inputPatterns, teacher, nodes, connections, \
    deltaArray, weightChangeArray, learningConstant, momentumConstant, numberOfHiddenNodes

    # Shuffle input:
    orders = list(range(0, len(inputPatterns)))
    random.shuffle(orders)
    sumPatternError = 0
    for index in orders:
        if teach:
            sumPatternError += runNetwork(inputPatterns[index], teacher[index])
        else:
            sumPatternError += runNetwork(inputPatterns[index], teacher[index], teach=False)
    populationError = sumPatternError/(len(nodes["output"]) * len(inputPatterns))

    # After epoch: momentumize weight changes
    for key in weightChangeArray.keys():
        for i in range(0, len(weightChangeArray[key])):
            for j in range(0, len(weightChangeArray[key][i])):
                weightChangeArray[key][i][j] = momentumConstant * weightChangeArray[key][i][j]

    return populationError

def testNetwork(showIO=True):
    if showIO:
        for i in range(0, len(inputPatterns)):
            runNetwork(inputPatterns[i], teacher[i], teach=False)
            print("Input: {}, Output: {}, Expected output: {}".format(inputPatterns[i], nodes["output"], teacher[i]))
    print("Population error: {}".format(runEpoch(teach=False)))

