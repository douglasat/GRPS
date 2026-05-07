#!/usr/bin/env python3.9
import numpy as np
from math import sqrt
from sklearn.preprocessing import normalize
import random


class HeuristicEstimation:
    def __init__(self, list_objects, list_parameters):
        self.listObjects = list_objects
        self.listParameters = list_parameters
        #self.listStates = list_states
        #self.X = []
        #self.Coefficients = []
        #self.listCost = list_cost

        #for a in range(len(list_states)):
        #    self.X.append(self.generateMatrix(list_states[a]))
        #    self.Coefficients.append(np.dot(np.linalg.pinv(np.dot(np.transpose(self.X[a]), self.X[a])),
        #                                    np.dot(np.transpose(self.X[a]), self.listCost[a])))

    # def generateMatrix(self, list_states):
    #     sample_matrix = np.zeros((len(list_states), 1 + len(self.listObjects) * len(self.listParameters)))
    #     for index_i in range(len(list_states)):
    #         # a = ' '.join(list_states[index_i])
    #         index_j = 1
    #         sample_matrix[index_i, 0] = 1
    #         for par in self.listParameters:
    #             for obj in self.listObjects:
    #                 count = 0
    #                 for a in list_states[index_i]:
    #                     b = set(list([par, obj]))
    #                     if b.issubset(set(a)):
    #                         count += 1
    #                 sample_matrix[index_i, index_j] = count
    #                 index_j += 1
    #
    #     return sample_matrix

    def generateMatrix_teste(self, list_states):
        sample_matrix = np.zeros((len(list_states), 1 + len(self.listObjects) * len(self.listParameters)))
        coefficients_objects = np.arange(2, len(self.listObjects) + 2, 1)
        for index_i in range(len(list_states)):
            index_j = 1
            sample_matrix[index_i, 0] = 1
            for par in self.listParameters:
                index_objects = 0
                for obj in self.listObjects:
                    count = 0
                    for a in list_states[index_i]:
                        b = set(list([par, obj]))
                        if b.issubset(set(a)):
                            count += 1
                    sample_matrix[index_i, index_j] = count*coefficients_objects[index_objects]
                    index_j += 1
                    index_objects += 1

        sample_matrix = normalize(sample_matrix)
        return sample_matrix

    def generateMatrix_teste(self, list_states):
        sample_matrix = np.zeros((len(list_states), 1 + len(self.listObjects) * len(self.listParameters)))

        coefficients_objects = np.linspace(0.1, 1, len(self.listObjects))
        dict_objs = dict(zip(self.listObjects, coefficients_objects))

        n = 0
        for state in list_states:
            dict_list = {}
            for joint in state:
                if dict_list.get(joint[0] + joint[1]) == None:
                    dict_list.update({joint[0] + joint[1]: 1})
                for object_joint in joint[1:]:
                    dict_list.update({joint[0] + joint[1]: dict_list.get(joint[0] + joint[1])*dict_objs.get(object_joint)})

            m = 0
            sample_matrix[n, m] = 1
            for i in range(len(self.listParameters)):
                for j in range((len(self.listObjects))):
                    if not dict_list.get(self.listParameters[i]+self.listObjects[j]) == None:
                        sample_matrix[n, m+1] = dict_list.get(self.listParameters[i]+self.listObjects[j])
                    m += 1
            n += 1

        return sample_matrix

    def generateMatrix1(self, list_states):
        sample_matrix = np.zeros((len(list_states), 1 + len(self.listObjects) * len(self.listParameters)))

        coefficients_objects = np.arange(2, len(self.listObjects) + 2, 1)
        dict_objs = dict(zip(self.listObjects, coefficients_objects))

        n = 0
        for state in list_states:
            dict_list = {}
            for joint in state:
                for object_joint in joint[1:]:
                    actual = joint[0] + object_joint
                    if dict_list.get(actual) == None:
                        dict_list.update({actual: dict_objs.get(object_joint)})
                    else:
                        dict_list.update({actual: dict_list.get(actual) + dict_objs.get(object_joint)})

            m = 0
            sample_matrix[n, m] = 1
            for i in range(len(self.listParameters)):
                for j in range(len(self.listObjects)):
                    if not dict_list.get(self.listParameters[i] + self.listObjects[j]) == None:
                        sample_matrix[n, m + 1] = dict_list.get(self.listParameters[i] + self.listObjects[j])
                    m += 1
            n += 1

        sample_matrix = normalize(sample_matrix)
        return sample_matrix


    def generateMatrix(self, list_states):
        sample_matrix = np.zeros((len(list_states), 1 + len(self.listObjects) * len(self.listParameters)))

        coefficients_objects = np.arange(2, len(self.listObjects) + 2, 1)
        dict_objs = dict(zip(self.listObjects, coefficients_objects))

        n = 0
        for state in list_states:
            dict_list = {}
            for joint in state:
                for object_joint in joint[1:]:
                    actual = joint[0] + object_joint
                    if dict_list.get(actual) == None:
                        dict_list.update({actual: dict_objs.get(object_joint)})
                    else:
                        dict_list.update({actual: dict_list.get(actual) + dict_objs.get(object_joint)})

            m = 0
            sample_matrix[n, m] = 1
            for i in range(len(self.listParameters)):
                for j in range(len(self.listObjects)):
                    if not dict_list.get(self.listParameters[i] + self.listObjects[j]) == None:
                        sample_matrix[n, m + 1] = dict_list.get(self.listParameters[i] + self.listObjects[j])
                    m += 1
            n += 1

        sample_matrix = sample_matrix
        return sample_matrix

    def calculateEstimation(self, list_states):
        sample_matrix = self.generateMatrix(list_states)
        return [np.dot(sample_matrix, a) for a in self.Coefficients]

    def updateEstimation(self, list_states, list_cost):
        for a in range(len(list_states)):
            self.X[a] = np.concatenate([self.X[a], self.generateMatrix(list_states[a])], axis=0)
            self.listCost[a] = np.concatenate((self.listCost[a], list_cost[a]), axis=None)
            self.Coefficients[a] = np.dot(np.linalg.pinv(np.dot(np.transpose(self.X[a]), self.X[a])),
                                          np.dot(np.transpose(self.X[a]), self.listCost[a]))
