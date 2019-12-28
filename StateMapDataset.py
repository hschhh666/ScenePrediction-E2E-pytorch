# -*- coding: utf-8 -*- 

'''
读取状态地图的数据接口

'''



import numpy as np
import os
import re
import torch
from torch.utils.data import Dataset
from cv2 import cv2
import random
import math

class FakeDataSet(Dataset):
    def __init__(self,EastGateFakeDataPath,train = True):
        self.train = train
        self.npyFiles = []
        for root,_,files in os.walk(EastGateFakeDataPath):
            for f in files:
                if re.match('.*npy',f):
                    npyFile = os.path.join(root,f)
                    self.npyFiles.append(npyFile) #get all npy file in current directory
            break
        if len(self.npyFiles) ==0:
            print('Error! There is no npy file in directory ',EastGateFakeDataPath)
            print('Program exit.')
            exit(-1)
        else:
            print('There are %d npy files in directory'%len(self.npyFiles))
            random.seed(666) #shuffle npy files, but making sure every time has the same value
            random.shuffle(self.npyFiles)

    def __len__(self):
        if self.train:
            lenth = len(self.npyFiles)*0.8# 80% data as training set
        else:
            lenth = len(self.npyFiles)*0.2# 20% data as testing set
        return int(lenth)
        

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if not self.train:
            idx = int(idx + len(self.npyFiles)*0.8)

        npyFile = self.npyFiles[idx]
        data = np.load(npyFile,allow_pickle=True)
        simulationTime = data[0]
        originPedestrianMatrix = data[1]
        generatedPedestrianMatrix = data[2]
        stateMap = data[3]
        stateMap = cv2.resize(stateMap,(512,512))
        stateMap = stateMap[np.newaxis,:]
        
        size = np.shape(generatedPedestrianMatrix)[0]
        pedestrianMatrix = np.zeros(size*size-size)

        count = 0
        for i in range(size):
            for j in range(size):
                if i==j:
                    continue
                pedestrianMatrix[count] = generatedPedestrianMatrix[i][j] #flatten matrix, but ignore diagonal element
                count = count + 1
        
        pedestrianMatrix = pedestrianMatrix.astype(np.float32)
        sample = {'stateMap':stateMap,'pedestrianMatrix':pedestrianMatrix}

        return sample


class FakeAvgDataset(Dataset):
    def __init__(self, E_path, SE_path, train = True):
        self.train = train
        self.E_path = E_path
        self.SE_path = SE_path

    def __len__(self):
        if self.train:
            return 80
        else:
            return 20

    def __getitem__(self,idx):
        idx += 1
        if not self.train:
            idx = idx + 80
        
        SE_npy = 'EastSouth_M' + str(idx)+'_AVG.npy'
        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = 'East_M' + str(idx) + '_AVG.npy'
        E_npy = os.path.join(self.E_path,E_npy)

        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)


        SEStateMap = SEData[3]
        SEStateMap = cv2.resize(SEStateMap,(512,512))
        SEStateMap = SEStateMap[np.newaxis,:]

        EStateMap = EData[3]
        EStateMap = cv2.resize(EStateMap,(512,512))
        EStateMap = EStateMap[np.newaxis,:]

        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap}

        return sample




class FakeSinglePairDataset(Dataset):
    def __init__(self,E_path,SE_path,train = True):
        self.train = train
        self.E_path = E_path
        self.SE_path = SE_path
        self.Psize = 5
        self.Msize = 20

    def __len__(self):
        return self.Psize*self.Msize
    
    def __getitem__(self,idx):
        M = int(idx / self.Psize)
        M = M + 1
        P = random.randint(0,self.Psize-1)
        if not self.train:
            P = P+self.Psize
        SE_npy = 'EastSouth_M%d_P%d.npy'%(M,P)
        P = random.randint(0,self.Psize-1)
        if not self.train:
            P = P+self.Psize
        E_npy = 'East_M%d_P%d.npy'%(M,P)

        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = os.path.join(self.E_path,E_npy)

        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)

        SEStateMap = SEData[3]
        SEStateMap = cv2.resize(SEStateMap,(512,512))
        SEStateMap = SEStateMap[np.newaxis,:]

        EStateMap = EData[3]
        EStateMap = cv2.resize(EStateMap,(512,512))
        EStateMap = EStateMap[np.newaxis,:]

        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap}

        return sample





def convertDataToBGR(datas):
    dim = len(np.shape(datas))

    def processSingleImg(data):
        toLeft = data[0]
        toRight = data[1]
        toUp = data[2]
        toDown = data[3]
        peopleNum = toLeft + toRight + toUp + toDown
        size = np.shape(toLeft)[0]

        peopleNumFigure = np.zeros([size,size,3])
        toLeftFigure = np.zeros([size,size,3])
        toRightFigure = np.zeros([size,size,3])
        toUpFigure = np.zeros([size,size,3])
        toDownFigure = np.zeros([size,size,3])

        visThreshold = 70
        singleVisThreshold = visThreshold / 2

        peopleNum[peopleNum > visThreshold] = visThreshold
        peopleNum = peopleNum * 255/visThreshold
        toLeft[toLeft > singleVisThreshold] = singleVisThreshold
        toLeft = toLeft * 255 / singleVisThreshold
        toRight[toRight > singleVisThreshold] = singleVisThreshold
        toRight = toRight * 255 / singleVisThreshold
        toUp[toUp > singleVisThreshold] = singleVisThreshold
        toUp = toUp * 255 / singleVisThreshold
        toDown[toDown > singleVisThreshold] = singleVisThreshold
        toDown = toDown * 255 /singleVisThreshold

        peopleNumFigure[:,:,0] = 255 - peopleNum
        peopleNumFigure[:,:,1] = 255 - peopleNum
        peopleNumFigure[:,:,2] = 255 - peopleNum
        
        toLeftFigure[:,:,0] = 255 - toLeft
        toLeftFigure[:,:,1] = 255 - toLeft
        toLeftFigure[:,:,2] = 255

        toRightFigure[:,:,0] = 255
        toRightFigure[:,:,1] = 255 - toRight
        toRightFigure[:,:,2] = 255 - toRight

        toUpFigure[:,:,0] = 255 - toUp
        toUpFigure[:,:,1] = ((255 - 92) * (255 - toUp) / 255)  + 92
        toUpFigure[:,:,2] = 255

        toDownFigure[:,:,0] = 255
        toDownFigure[:,:,1] = 255 - toDown
        toDownFigure[:,:,2] = ((255 - 158) * (255 - toDown) / 255)  + 158

        peopleNumFigure = peopleNumFigure.astype(np.uint8)
        toLeftFigure = toLeftFigure.astype(np.uint8)
        toRightFigure = toRightFigure.astype(np.uint8)
        toUpFigure = toUpFigure.astype(np.uint8)
        toDownFigure = toDownFigure.astype(np.uint8)

        bgr = np.zeros([size,size*5,3])
        bgr[:,size*0:size*1,:] = toUpFigure
        bgr[:,size*1:size*2,:] = toDownFigure
        bgr[:,size*2:size*3,:] = peopleNumFigure
        bgr[:,size*3:size*4,:] = toLeftFigure
        bgr[:,size*4:size*5,:] = toRightFigure
        return bgr

    if dim == 3: 
        bgr = processSingleImg(datas)
        return bgr
    
    if dim == 4:
        datas = datas.numpy()
        bgrs = []
        for k in range(np.shape(datas)[0]):
            bgr = processSingleImg(datas[k])
            bgrs.append(bgr)
        bgrs = np.array(bgrs)
        bgrs = np.transpose(bgrs,(0,3,1,2))
        bgrs = torch.Tensor(bgrs)
        return bgrs

    else:
        print('dim error! check data dim.(dim should be 3 or 4)')
        exit(-1)


    



class subDatasetIndex():
    def __init__(self):
        eastIndex = [i*15 + j for i in range(54) for j in [1,5,10] ]
        southEastIndex = [i*15 + j for i in range(54) for j in [1,5,10] ]
        pass

class typicalTestData(Dataset):
    def __init__(self,E_path,SE_path):
        self.E_path = E_path
        self.SE_path = SE_path
        self.deltaT = 0
        self.idx = [60,180,300,600] # all in ; no one ; all out ; mid
        # M = [4,12,20,40]
        # self.idx = [25,55,85,115]
    
    def __len__(self):
        return 4

    def __getitem__(self,idx):
        E_npy = 'East_M%d_P0.npy'%(self.idx[idx])
        SE_npy = 'EastSouth_M%d_P0.npy'%(self.idx[idx])
        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = os.path.join(self.E_path,E_npy)
        
        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)

        toRight = SEData[5]
        toRight = cv2.resize(toRight,(512,512))
        toRight = toRight[np.newaxis,:]
        toLeft = SEData[4]
        toLeft = cv2.resize(toLeft,(512,512))
        toLeft = toLeft[np.newaxis,:]
        SEStateMap = np.concatenate((toLeft,toRight))

        toRight = EData[5]
        toRight = cv2.resize(toRight,(512,512))
        toRight = toRight[np.newaxis,:]
        toLeft = EData[4]
        toLeft = cv2.resize(toLeft,(512,512))
        toLeft = toLeft[np.newaxis,:]
        EStateMap = np.concatenate((toLeft,toRight))

        # bgr = convertDataToBGR(EStateMap)
        # cv2.imwrite('/home/hsc/test.jpg',bgr)
        self.deltaT = np.array(self.deltaT,dtype = np.float32)
        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap,'deltaT':self.deltaT}
        return sample


class FakeDeltaTDataset(Dataset):
    def __init__(self,E_path,SE_path,deltaT,train = True):
        self.train = train
        self.E_path = E_path
        self.SE_path = SE_path
        eastIndex = [i*30 + j for i in range(24) for j in [1,10,20] ]
        southEastIndex = [i*60 + 30 + j for i in range(12) for j in [1,10,20] ]
        # eastIndex = [i*30 + j for i in range(27) for j in [23,24,25,26,27] ]
        # southEastIndex = [i*30 + j for i in range(27) for j in [23,24,25,26,27] ]
        eastIndex = np.array(eastIndex)
        southEastIndex = np.array(southEastIndex)

        self.deltaT = deltaT
        self.TimeInterval = 30
        self.eastIndex = []
        self.southEastIndex = southEastIndex

        for i in eastIndex:
            paired = False
            M = int((i-1)/self.TimeInterval) + 1
            for j in [M + deltaT, M - deltaT]:
                startIdx = (j-1)*self.TimeInterval + 1
                southEastTmpIdx = southEastIndex[np.logical_and(startIdx <= southEastIndex,southEastIndex <= (startIdx + self.TimeInterval-1) )]
                paired = True if len(southEastTmpIdx)>0 else False
                if paired:
                    self.eastIndex.append(i)
                    break
        
        self.eastIndex = np.array(self.eastIndex)
        pass

    def __len__(self):
        if self.train:
            return len(self.eastIndex)
        else:
            return 3*24


    def __getitem__(self,idx):

        resultSouthEastIdx = resultEastIdx = 0
        if not self.train:
            testingIndex = [i*30 + j for i in range(24) for j in [2,11,21] ]
            resultSouthEastIdx = resultEastIdx = testingIndex[idx]

        else:
            resultEastIdx = self.eastIndex[idx]
            resultSouthEastIdx = -1

            M = int((resultEastIdx-1)/self.TimeInterval) + 1
            M = [M + self.deltaT, M - self.deltaT]
            random.shuffle(M)
            for m in M:
                startIdx = (m-1)*self.TimeInterval + 1
                southEastTmpIdx = self.southEastIndex[np.logical_and(startIdx <= self.southEastIndex,self.southEastIndex <= (startIdx + self.TimeInterval - 1) )]
                if len(southEastTmpIdx) >0:
                    random.shuffle(southEastTmpIdx)
                    resultSouthEastIdx = southEastTmpIdx[0]
                    break
        

        E_npy = 'East_M%d_P0.npy'%(resultEastIdx)
        SE_npy = 'EastSouth_M%d_P0.npy'%(resultSouthEastIdx)
        SE_npy = os.path.join(self.SE_path,SE_npy)
        E_npy = os.path.join(self.E_path,E_npy)
        
        SEData = np.load(SE_npy,allow_pickle=True)
        EData = np.load(E_npy,allow_pickle=True)


        toRight = SEData[5]
        toRight = cv2.resize(toRight,(512,512))
        toRight = toRight[np.newaxis,:]
        toLeft = SEData[4]
        toLeft = cv2.resize(toLeft,(512,512))
        toLeft = toLeft[np.newaxis,:]
        toUp = SEData[6]
        toUp = cv2.resize(toUp,(512,512))
        toUp = toUp[np.newaxis,:]
        toDown = SEData[7]
        toDown = cv2.resize(toDown,(512,512))
        toDown = toDown[np.newaxis,:]
        SEStateMap = np.concatenate((toLeft,toRight,toUp,toDown))
        
        toRight = EData[5]
        toRight = cv2.resize(toRight,(512,512))
        toRight = toRight[np.newaxis,:]
        toLeft = EData[4]
        toLeft = cv2.resize(toLeft,(512,512))
        toLeft = toLeft[np.newaxis,:]
        toUp = EData[6]
        toUp = cv2.resize(toUp,(512,512))
        toUp = toUp[np.newaxis,:]
        toDown = EData[7]
        toDown = cv2.resize(toDown,(512,512))
        toDown = toDown[np.newaxis,:]
        EStateMap = np.concatenate((toLeft,toRight,toUp,toDown))

        # bgr = convertDataToBGR(EStateMap)
        # cv2.imwrite('/home/hsc/test.jpg',bgr)

        self.deltaT = np.array(self.deltaT,dtype = np.float32)
        sample = {'EStateMap':EStateMap,'SEStateMap':SEStateMap,'deltaT':self.deltaT}
        return sample

        


if __name__ == '__main__':
    # test = FakeDataSet('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate')
    # sample = test[0]
    # a,b = sample['stateMap'], sample['pedestrianMatrix']
    # print('stateMap size = ',np.shape(a))
    # print('pedestrianMatrix size = ',np.shape(b))

    # maxValue = np.max(a)
    # a = 255 - (a/maxValue)*255
    # cv2.imwrite('/home/hsc/test.jpg',a[0,:,:])



    # test = FakeAvgDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/AvgStateMap','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/AvgStateMap')
    # sample = test[70]
    # E, SE = sample['EStateMap'], sample['SEStateMap']
    
    # maxValue = np.max(E)
    # E = 255 - (E/maxValue)*255
    # cv2.imwrite('/home/hsc/E.jpg',E[0,:,:])

    # maxValue = np.max(SE)
    # SE = 255 - (SE/maxValue)*255
    # cv2.imwrite('/home/hsc/SE.jpg',SE[0,:,:])

    # test = FakeSinglePairDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data2','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data2')
    # print(test[3])

    

    filename = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data4/East_M111_P0.npy'
    data = np.load(filename,allow_pickle=True)

    toRight = data[5]
    toRight = cv2.resize(toRight,(512,512))
    toRight = toRight[np.newaxis,:]
    toLeft = data[4]
    toLeft = cv2.resize(toLeft,(512,512))
    toLeft = toLeft[np.newaxis,:]
    data = np.concatenate((toLeft,toRight))
    data = np.array(data)
    bgr = convertDataToBGR(data)
    cv2.imwrite('/home/hsc/test1.jpg',bgr)



    test = FakeDeltaTDataset('/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data4','/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data4',1)

    for i in range(5*26):
        test[i]

    pass
