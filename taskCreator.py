import random
import copy

def createRandomTask(inputNodes, hiddenNodes, outputNodes, populationSize, auto, learningConstant=0.1,
                     momentumConstant=0.9):
    inputList = []
    while len(inputList) < populationSize:
        new_list = []
        for j in range(0, inputNodes):
            newList.append(random.randint(0,1))
        if in
        inputList.append(new_list)

    # if autoassociative, input = output, otherwise output is random
    outputList = []
    if auto:
        outputList = copy.deepcopy(inputList)
    else:
        for i in range(0, populationSize):
            outputList.append([])
            for j in range(0, outputNodes):
                outputList[i].append(random.randint(0, 1))


    task = {}
    task["inputPatterns"] = inputList
    task["learningConstant"] = learningConstant
    task["momentumConstant"] = momentumConstant
    task["numberOfHiddenNodes"] = hiddenNodes
    task["teacher"] = outputList

    return task

def printTask(task):
    print("Learning constant: ", task["learningConstant"])
    print("Momentum constant: ", task["momentumConstant"])
    print("Input patterns: ]")
    for row in range(0, len(task["inputPatterns"])):
        print("{}".format(task["inputPatterns"][row]))
    print("]")
    print("Teacher patterns: [")
    for row in range(0, len(task["teacher"])):
        print("{}".format(task["teacher"][row]))
    print("]")
