import neuralnetwork as nn
from tqdm import tqdm
import time
import sys
import tasks as tc

tasks = {}

# tasks["EasyRandom"] = {
#     "learningConstant": 0.1,
#     "momentumConstant": 0.9,
#     "numberOfHiddenNodes": 3,
#
#     "inputPatterns": [
#         [1, 1, 1, 0, 0],
#         [0, 0, 1, 1, 1]
#     ],
#
#     "teacher": [
#         [1, 1, 0, 0, 1],
#         [1, 0, 0, 1, 1]
#     ]
# }

# tasks["XOR"] = {
#     "learningConstant": 0.1,
#     "momentumConstant": 0.9,
#     "numberOfHiddenNodes": 2,
#
#     "inputPatterns": [
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
#     ],
#
#     "teacher": [
#         [0],
#         [1],
#         [1],
#         [0]
#     ]
# }
#
# tasks["8bitEncoder"] = {
#     "learningConstant": 0.1,
#     "momentumConstant": 0.9,
#     "numberOfHiddenNodes": 3,
#
#     "inputPatterns": [
#         [0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0]
#     ],
#
#     "teacher": [
#         [0, 0, 0, 0, 0, 0, 0, 1],
#         [0, 0, 0, 0, 0, 0, 1, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0]
#     ]
# }
#
# tasks["3bitParity"] = {
#     "learningConstant": 0.1,
#     "momentumConstant": 0.9,
#     "numberOfHiddenNodes": 3,
#
#     "inputPatterns": [
#         [1, 1, 1],
#         [1, 1, 0],
#         [1, 0, 1],
#         [0, 1, 1],
#         [0, 0, 0],
#         [0, 0, 1],
#         [0, 1, 0],
#         [1, 0, 0]
#     ],
#
#     "teacher": [
#         [0],
#         [1],
#         [1],
#         [1],
#         [1],
#         [0],
#         [0],
#         [0]
#     ]
# }
#
# tasks["4bitParity"] = {
#     "learningConstant": 0.1,
#     "momentumConstant": 0.9,
#     "numberOfHiddenNodes": 4,
#
#     "inputPatterns": [
#         [1, 1, 1, 1],
#         [1, 1, 1, 0],
#         [1, 1, 0, 1],
#         [1, 0, 1, 1],
#         [0, 1, 1, 1],
#         [1, 1, 0, 0],
#         [1, 0, 1, 0],
#         [1, 0, 0, 1],
#         [0, 0, 1, 1],
#         [0, 1, 0, 1],
#         [0, 1, 1, 0],
#         [0, 0, 0, 0],
#         [0, 0, 0, 1],
#         [0, 0, 1, 0],
#         [0, 1, 0, 0],
#         [1, 0, 0, 0],
#     ],
#
#     "teacher": [
#         [1],
#         [0],
#         [0],
#         [0],
#         [0],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [1],
#         [0],
#         [0],
#         [0],
#         [0]
#     ]
# }
#
# # tasks["iris"] = {
# #     "learningConstant": 0.1,
# #     "momentumConstant": 0.9,
# #     "numberOfHiddenNodes": 3,
# #
# #     "inputPatterns": [
# #         [0.224, 0.624, 0.067, 0.043],
# #         [0.11, 0.502, 0.051, 0.043],
# #         [0.196, 0.667, 0.067, 0.043],
# #         [0.055, 0.584, 0.067, 0.082],
# #         [0.027, 0.376, 0.067, 0.043],
# #         [0.306, 0.71, 0.086, 0.043],
# #         [0.137, 0.416, 0.067, 0.0],
# #         [0.416, 0.831, 0.035, 0.043],
# #         [0.306, 0.792, 0.051, 0.125],
# #         [0.388, 0.749, 0.118, 0.082],
# #         [0.306, 0.584, 0.118, 0.043],
# #         [0.082, 0.667, 0.0, 0.043],
# #         [0.137, 0.584, 0.153, 0.043],
# #         [0.196, 0.584, 0.102, 0.125],
# #         [0.251, 0.584, 0.067, 0.043],
# #         [0.137, 0.459, 0.102, 0.043],
# #         [0.251, 0.875, 0.086, 0.0],
# #         [0.165, 0.459, 0.086, 0.043],
# #         [0.333, 0.624, 0.051, 0.043],
# #         [0.027, 0.416, 0.051, 0.043],
# #         [0.196, 0.624, 0.051, 0.082],
# #         [0.027, 0.502, 0.051, 0.043],
# #         [0.224, 0.749, 0.153, 0.125],
# #         [0.224, 0.749, 0.102, 0.043],
# #         [0.278, 0.71, 0.086, 0.043],
# #         [0.165, 0.416, 0.067, 0.043],
# #         [0.082, 0.459, 0.086, 0.043],
# #         [0.306, 0.792, 0.118, 0.125],
# #         [0.196, 0.584, 0.086, 0.043],
# #         [0.165, 0.459, 0.086, 0.0],
# #         [0.137, 0.584, 0.102, 0.043],
# #         [0.0, 0.416, 0.016, 0.0],
# #         [0.388, 1.0, 0.086, 0.125],
# #         [0.224, 0.624, 0.067, 0.082],
# #         [0.224, 0.749, 0.086, 0.082],
# #         [0.224, 0.71, 0.086, 0.125],
# #         [0.224, 0.541, 0.118, 0.165],
# #         [0.196, 0.416, 0.102, 0.043],
# #         [0.251, 0.624, 0.086, 0.043],
# #         [0.11, 0.502, 0.102, 0.043],
# #         [0.306, 0.584, 0.086, 0.125],
# #         [0.333, 0.918, 0.067, 0.043],
# #         [0.196, 0.502, 0.035, 0.043],
# #         [0.165, 0.667, 0.067, 0.0],
# #         [0.224, 0.584, 0.086, 0.043],
# #         [0.055, 0.125, 0.051, 0.082],
# #         [0.196, 0.624, 0.102, 0.208],
# #         [0.137, 0.416, 0.067, 0.082],
# #         [0.082, 0.502, 0.067, 0.043],
# #         [0.196, 0.541, 0.067, 0.043],
# #         [0.749, 0.502, 0.627, 0.541],
# #         [0.722, 0.459, 0.663, 0.584],
# #         [0.612, 0.333, 0.612, 0.584],
# #         [0.557, 0.541, 0.627, 0.624],
# #         [0.639, 0.376, 0.612, 0.498],
# #         [0.196, 0.0, 0.424, 0.376],
# #         [0.471, 0.082, 0.51, 0.376],
# #         [0.361, 0.376, 0.439, 0.498],
# #         [0.361, 0.416, 0.592, 0.584],
# #         [0.529, 0.082, 0.592, 0.584],
# #         [0.443, 0.502, 0.643, 0.459],
# #         [0.557, 0.208, 0.663, 0.584],
# #         [0.584, 0.376, 0.561, 0.498],
# #         [0.694, 0.333, 0.643, 0.541],
# #         [0.471, 0.376, 0.592, 0.584],
# #         [0.333, 0.169, 0.475, 0.416],
# #         [0.416, 0.29, 0.49, 0.459],
# #         [0.306, 0.416, 0.592, 0.584],
# #         [0.667, 0.459, 0.627, 0.584],
# #         [0.361, 0.416, 0.525, 0.498],
# #         [0.333, 0.251, 0.576, 0.459],
# #         [0.416, 0.251, 0.51, 0.459],
# #         [0.361, 0.29, 0.541, 0.498],
# #         [0.388, 0.376, 0.541, 0.498],
# #         [0.224, 0.208, 0.337, 0.416],
# #         [0.584, 0.502, 0.592, 0.584],
# #         [0.333, 0.125, 0.51, 0.498],
# #         [0.388, 0.333, 0.592, 0.498],
# #         [0.165, 0.169, 0.388, 0.376],
# #         [0.251, 0.29, 0.49, 0.541],
# #         [0.443, 0.416, 0.541, 0.584],
# #         [0.498, 0.376, 0.627, 0.541],
# #         [0.667, 0.459, 0.576, 0.541],
# #         [0.416, 0.29, 0.525, 0.376],
# #         [0.361, 0.208, 0.49, 0.416],
# #         [0.498, 0.333, 0.51, 0.498],
# #         [0.498, 0.333, 0.627, 0.459],
# #         [0.639, 0.416, 0.576, 0.541],
# #         [0.667, 0.416, 0.678, 0.667],
# #         [0.388, 0.251, 0.424, 0.376],
# #         [0.333, 0.169, 0.459, 0.376],
# #         [0.471, 0.29, 0.694, 0.624],
# #         [0.471, 0.584, 0.592, 0.624],
# #         [0.557, 0.125, 0.576, 0.498],
# #         [0.333, 0.208, 0.51, 0.498],
# #         [0.498, 0.416, 0.612, 0.541],
# #         [0.196, 0.125, 0.388, 0.376],
# #         [0.388, 0.416, 0.541, 0.459],
# #         [0.529, 0.376, 0.561, 0.498],
# #         [0.388, 0.333, 0.525, 0.498],
# #         [0.557, 0.541, 0.847, 1.0],
# #         [0.776, 0.416, 0.831, 0.831],
# #         [0.612, 0.416, 0.812, 0.875],
# #         [0.165, 0.208, 0.592, 0.667],
# #         [0.667, 0.208, 0.812, 0.71],
# #         [0.612, 0.502, 0.694, 0.792],
# #         [0.694, 0.416, 0.761, 0.831],
# #         [0.416, 0.333, 0.694, 0.957],
# #         [0.612, 0.416, 0.761, 0.71],
# #         [0.945, 0.251, 1.0, 0.918],
# #         [0.722, 0.502, 0.796, 0.918],
# #         [0.945, 0.333, 0.965, 0.792],
# #         [0.667, 0.541, 0.796, 0.831],
# #         [0.529, 0.333, 0.643, 0.71],
# #         [0.584, 0.333, 0.78, 0.831],
# #         [0.863, 0.333, 0.863, 0.749],
# #         [0.584, 0.333, 0.78, 0.875],
# #         [0.498, 0.251, 0.78, 0.541],
# #         [0.557, 0.584, 0.78, 0.957],
# #         [0.471, 0.416, 0.643, 0.71],
# #         [0.667, 0.459, 0.78, 0.957],
# #         [0.416, 0.29, 0.694, 0.749],
# #         [0.667, 0.541, 0.796, 1.0],
# #         [0.557, 0.208, 0.678, 0.749],
# #         [0.529, 0.584, 0.745, 0.918],
# #         [0.416, 0.29, 0.694, 0.749],
# #         [0.557, 0.376, 0.78, 0.71],
# #         [0.918, 0.416, 0.949, 0.831],
# #         [0.835, 0.376, 0.898, 0.71],
# #         [0.804, 0.667, 0.863, 1.0],
# #         [0.584, 0.29, 0.729, 0.749],
# #         [0.388, 0.208, 0.678, 0.792],
# #         [0.584, 0.502, 0.729, 0.918],
# #         [0.945, 0.749, 0.965, 0.875],
# #         [0.471, 0.082, 0.678, 0.584],
# #         [0.361, 0.333, 0.663, 0.792],
# #         [0.557, 0.29, 0.663, 0.71],
# #         [0.804, 0.502, 0.847, 0.71],
# #         [0.498, 0.416, 0.51, 0.71],
# #         [0.804, 0.416, 0.812, 0.624],
# #         [1.0, 0.749, 0.914, 0.792],
# #         [0.557, 0.333, 0.694, 0.584],
# #         [0.945, 0.416, 0.863, 0.918],
# #         [0.584, 0.459, 0.761, 0.71],
# #         [0.722, 0.459, 0.745, 0.831],
# #         [0.722, 0.459, 0.694, 0.918],
# #         [0.694, 0.502, 0.831, 0.918],
# #         [0.667, 0.416, 0.714, 0.918],
# #         [0.612, 0.416, 0.714, 0.792],
# #         [0.443, 0.416, 0.694, 0.71]
# #     ],
# #
# #     "teacher": [
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [1.0, 0.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 1.0, 0.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #         [0.0, 0.0, 1.0],
# #     ]
# # }


task = (tc.createRandomTask(inputNodes=512, outputNodes=512, hiddenNodes=128, populationSize=10, auto=True, learningConstant=0.3,
                            momentumConstant=0.5))
tc.printTask(task)
tasks["random"] = task


"""
Testing
"""
maxEpochs = 10000
taskRepeats = 200
errorThreshold = 0.03

# Logging
log = False
outfile = None
if log:
    mytime = time.gmtime()
    filename = "log:{}-{}-{}--{}:{}:{}.txt".format(mytime[0], mytime[1], mytime[2], mytime[3], mytime[4], mytime[5])
    outfile = open(filename, 'w')

taskKeys = tasks.keys()


for key in taskKeys: # For each task
    sys.stdout.flush()
    print("Beginning {} test".format(key))
    if log:
        outfile.write("Beginning {} test\n".format(key))
        outfile.flush()
    task = tasks[key]
    success = 0
    failure = 0
    sumEpochs = 0
    for taskRepeat in tqdm(range(0, taskRepeats)): # For the repeat of each task
        nn.initalise(input=task["inputPatterns"], teach=task["teacher"], momentum=task["momentumConstant"],
                     hiddenNodes=task["numberOfHiddenNodes"], learning=task["learningConstant"])
        j = 0
        for j in range(0, maxEpochs):
            error = nn.runEpoch()
            if error < errorThreshold:
                break
        if nn.runEpoch(teach=False) < errorThreshold:
            success += 1
        else:
            failure += 1
        sumEpochs += j
        # print("Task {} epochs taken: {}".format(taskIndex+1, j+1))

    # nn.testNetwork()
    sys.stdout.flush()
    toPrint = "\nTests: {}, Success rate: {}% ({} successful, {} failed. {} average epochs.)".format(
        taskRepeats, (success / (taskRepeats)) * 100, success, failure, round(sumEpochs/taskRepeats, 0))
    nn.testNetwork()

    print(toPrint)

    if log:
        outfile.flush()
        outfile.write(toPrint+"\n")
    sys.stdout.flush()
