#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 20:07:40 2022

@author: dom
"""
import numpy as np
import data_utils as du
import scipy as sc


def resortArray(body,mapB25,mapH99):
    #starting here for test - array with 96 zeroes
    testArray = np.zeros(96)
    #math for point 13 of h99, a point between 1 and 14, or 8 and 1 using body25
    testArray[13*3] = body25[1][0] - body25[8][0]
    testArray[13*3+1] = body25[1][1] - body25[8][1]
    testArray[13*3+2] = body25[1][2] - body25[8][2]
    #TODO: join 16 in H99, somewhere between 15 and 16 in B25
    #fill the array with the data we know from Evans model, aside from points 13 and 16
    for i in range(len(mapIndexB25)):
        #how it is indexed at line 310 forward_kinematics.py
        x = mapIndexH99[i]*3
        y = x + 1
        z = y + 1
        #index return array using these
        testArray[x] = body25[mapIndexB25[i]][0]
        testArray[y] = body25[mapIndexB25[i]][1]
        testArray[z] = body25[mapIndexB25[i]][2]
    
    
    testArray = np.reshape(testArray, (32,-1))
    testArray = testArray[:,[0,2,1]]
    
    return testArray

def pointsToVector(body,parent):
    retVal =[]
    for i in range(len(body)):
        if parent[i] == -1:
            #base position (pelvis), not sure what to do here, for now we say vector is 0
            retVal.append([0,0,0])
        else:
            child = body[i]
            parent = body[parent[i]]
            result = parent-child
            divisor = np.linalg.norm(result)
            retVal.append(result/divisor)
    return retVal
        
def vectorToRotMat(position1,position2):
    retVal = []
    for i in range(len(position1)):
        #TODO: this could be an error, not sure if align_vectors is the right function for the job
        val = sc.spacial.transform.Rotation.align_vectors(position1[i],position2[i])
        retVal.append(val.as_matrix())
    return retVal
    
    
    


def expMapFromBody(position1,position2):
    #define static variables
    mapB25 = [8,9,10,11,12,13,14,1,0,5,6,7,2,3,4]
    mapH99 = [1,2,3,4,7,8,9,14,15,26,27,28,18,19,20]-1
    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                      17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1
    #two positions, body25 = len 26 of xyz position, resort them to follow output of fkl
    position1 = resortArray(position1, mapB25, mapH99)
    position2 = resortArray(position2, mapB25, mapH99)
    #turn each position into a unit vector
    position1 = pointsToVector(position1, parent)
    position2 = pointsToVector(position2, parent)
    #calculate rotmat
    finalRotMatrix = vectorToRotMat(position1,position2)#import 'human-motion-prediction-pytorch-uncertainty/src/data_utils.py'
    #Hooray we have the rotation matricies. Now convert them to ExpMap
    finalRotMatrix = [lambda x: du.rotmat2expmap(x) for x in finalRotMatrix]
    #now create 99 long with correct positioning and lots of 0s :)
    returnVal = np.zeros(99)
    #fill it with the places
    expmapInd = np.split(np.arange(4,100)-1,32)
    for i in range(32):
        returnVal[expmapInd[i]] = finalRotMatrix[i]
    return returnVal
    
    
    
    
    
    
    
    



def declareVar():
    #TODO: deal with index 13 and 16 for H99
    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9,10, 1,12,13,14,15,13,
                      17,18,19,20,21,20,23,13,25,26,27,28,29,28,31])-1
    mapIndexB25 = [8,9,10,11,12,13,14,1,0,5,6,7,2,3,4]
    mapIndexH99 = [1,2,3,4,7,8,9,14,15,26,27,28,18,19,20]
    offset = np.array([0.000000,0.000000,0.000000,-132.948591,0.000000,0.000000,0.000000,-442.894612,0.000000,0.000000,-454.206447,0.000000,0.000000,0.000000,162.767078,0.000000,0.000000,74.999437,132.948826,0.000000,0.000000,0.000000,-442.894413,0.000000,0.000000,-454.206590,0.000000,0.000000,0.000000,162.767426,0.000000,0.000000,74.999948,0.000000,0.100000,0.000000,0.000000,233.383263,0.000000,0.000000,257.077681,0.000000,0.000000,121.134938,0.000000,0.000000,115.002227,0.000000,0.000000,257.077681,0.000000,0.000000,151.034226,0.000000,0.000000,278.882773,0.000000,0.000000,251.733451,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999627,0.000000,100.000188,0.000000,0.000000,0.000000,0.000000,0.000000,257.077681,0.000000,0.000000,151.031437,0.000000,0.000000,278.892924,0.000000,0.000000,251.728680,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,99.999888,0.000000,137.499922,0.000000,0.000000,0.000000,0.000000])
    offset = offset.reshape(-1,3)
    
    
    rotInd = [[5, 6, 4],
                [8, 9, 7],
                [11, 12, 10],
                [14, 15, 13],
                [17, 18, 16],
                [],
                [20, 21, 19],
                [23, 24, 22],
                [26, 27, 25],
                [29, 30, 28],
                [],
                [32, 33, 31],
                [35, 36, 34],
                [38, 39, 37],
                [41, 42, 40],
                [],
                [44, 45, 43],
                [47, 48, 46],
                [50, 51, 49],
                [53, 54, 52],
                [56, 57, 55],
                [],
                [59, 60, 58],
                [],
                [62, 63, 61],
                [65, 66, 64],
                [68, 69, 67],
                [71, 72, 70],
                [74, 75, 73],
                [],
                [77, 78, 76],
                []]

    
    return mapIndexB25,mapIndexH99, offset, rotInd




def reversefkl(body25, mapIndexB25, mapIndexH99, offset):
    #assert it is the correct array
    assert(len(body25) == 26)
    #declare variables
    returnValue = np.zeros((99))
    rootValue = np.array(body25[8])
    rootedArray = []
    """
    #TODO: This is wrong - Ask Justin - re-root the array in the correct position 
    for i in body25:
        rootedArray.append(i-rootValue)
    #re-organize the body25 indicies to align with the h99
    """
    
    
    #flatten the array: len = 75
    #flattenedArray = np.reshape(rootedArray, -1)
    #ensure it is a flat array (not (1,75), just (75))
    #flattenedArray = flattenedArray.squeeze()
    
    
    #starting here for test - array with 96 zeroes
    testArray = np.zeros(96)
    
    #math for point 13 of h99, a point between 1 and 14, or 8 and 1 using body25
    testArray[13*3] = body25[1][0] - body25[8][0]
    testArray[13*3+1] = body25[1][1] - body25[8][1]
    testArray[13*3+2] = body25[1][2] - body25[8][2]
    

    
    
    
    
    #fill the array with the data we know from Evans model, aside from points 13 and 16
    for i in range(len(mapIndexB25)):
        #how it is indexed at line 310 forward_kinematics.py
        x = mapIndexH99[i]*3
        y = x + 1
        z = y + 1
        #index return array using these
        testArray[x] = body25[mapIndexB25[i]][0]
        testArray[y] = body25[mapIndexB25[i]][1]
        testArray[z] = body25[mapIndexB25[i]][2]
        
    #this is when it is returned from fkl
    testArray = testArray.reshape(32,3)
    
    #arbitrarily reshape :)
    testArray = testArray[:,[0,2,1]]
    
    
    #begin changing
    
    njoints   = 32
    xyzStruct = [dict() for x in range(njoints)]
    
    
    for i in range(32):
        r = angles[ expmapInd[i] ]
        if parent[i] == -1: # we are at root node
            xyzStruct[i]['rotation'] = 
    
    
    
    
    '''
    #math for point 13 of h99, a point between 1 and 14, or 8 and 1 using body25
    testArray[] = body25[1][0] - body25[8][0]
    index13y = body25[1][1] - body25[8][1]
    index13z = body25[1][2] - body25[8][2]
    '''
    '''
    count = 0
    for i in testArray:
        if(i != 0):
            count += 1
    print(count)
    '''
    return testArray



mapIndexB25, mapIndexH99, offset, rotInd = declareVar()
body25 = np.arange(1,79).reshape(26,3)
retVal = reversefkl(body25, mapIndexB25, mapIndexH99, offset)
print(retVal)
retVal = retVal.reshape(-1)
dimsIgnored= [10, 11, 16, 17, 18, 19, 20, 25, 26, 31, 32, 33, 34, 35, 48, 49, 50, 58, 59, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
for i in dimsIgnored:
    assert retVal[i]==0, i
    
    
    