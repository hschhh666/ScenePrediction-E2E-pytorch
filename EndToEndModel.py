# -*- coding: utf-8 -*- 

'''
程序说明：东门和东南门各有自己的自编码器网络，开始实现算法，使用单次仿真数据而非平均数据
'''

import numpy as np
from cv2 import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import time
from datetime import datetime
from StateMapDataset import  FakeDeltaTDataset, typicalTestData, convertDataToBGR
import os,sys
from logger import Logger
from AutoEncoder import BehaviorModelAutoEncoder
import itertools
from tensorboardX import SummaryWriter
import argparse

if __name__ == '__main__':


    argParser = argparse.ArgumentParser(description='python arguments')
    argParser.add_argument('-cuda',type=int ,help='cuda device id')
    argParser.add_argument('-zdim',type=int,help='z dimention')
    argParser.add_argument('-dropout',type=float ,help='dropout p',default=0)
    args = argParser.parse_args()
    
    if args.cuda == None or args.zdim == None:
        print('[Error] No parameter. Program exit')
        exit(-2)
    if args.cuda < 0 or args.cuda >= torch.cuda.device_count():
        print('cuda %d does not exit! Program exit'%args.cuda)
        exit(-2)
    if args.zdim <= 0:
        print('z dim cannot <= zero! Program exit')
        exit(-2)
    if args.dropout >=1 or args.dropout <0:
        print('dropout p should be [0,1). Program exit')
        exit(-2)

    TestOrTrain = 'train'
    saveThisExper = False

    E_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/EastGate/data5'
    SE_dataset_path = '/home/hsc/Research/StateMapPrediction/datas/fake/SouthEastGate/data5'


    device = 'cuda:' + str(args.cuda)
    device = torch.device(device)

    if TestOrTrain =='train':

        # 当前路径下的resultDir用来保存每次的试验结果，包括log、结果图、训练参数。每次实验都在resultDir下创建一个以实验开始时间为名字的文件夹，该文件夹下保存当次实验的所有结果。
        # 如果resultDir不存在，则创建
        curPath = os.path.split(os.path.realpath(__file__))[0]
        resultDir = 'resultDir'
        resultDir = os.path.join(curPath,resultDir)
        if not os.path.exists(resultDir):
            print('create result dir')
            os.makedirs(resultDir)

        # 获取实验开始时间，并在resultDir下创建以该时间为名字的文件夹，用以保存本次实验结果
        curTime = time.strftime("%Y%m%d_%H_%M_%S", time.localtime())
        if saveThisExper:
            resultDir = os.path.join(resultDir,curTime)
            os.makedirs(resultDir)
        else:
            resultDir = os.path.join(resultDir,'tmp')
            resultDir = os.path.join(resultDir,curTime)
            os.makedirs(resultDir)

        tbLogDir = os.path.join(resultDir,'tbLog')
        tensorboardWriter = SummaryWriter(logdir=tbLogDir,flush_secs=1)

        # 创建log文件、img文件夹和modelParam文件夹，分别表示本次实验的日志、实验结果存储文件夹和模型参数存储文件夹
        logfileName = os.path.join(resultDir,curTime+'.txt')
        sys.stdout = Logger(logfileName)
        imgFolder = os.path.join(resultDir,'img')
        os.makedirs(imgFolder)
        modelParamFolder = os.path.join(resultDir,'modelParam')
        os.makedirs(modelParamFolder)

        # 加载数据集
        fakeSingleTrainsets = [FakeDeltaTDataset(E_dataset_path,SE_dataset_path,i,train = True) for i in range(-4,5)]
        
        fakeSingleTestset = FakeDeltaTDataset(E_dataset_path,SE_dataset_path,0,train = False)
        fakeSingleTestLoader = DataLoader(fakeSingleTestset,batch_size=4,shuffle=True)
        
        print('device = ',device)
        print('z-dim = ',args.zdim)
        print('dropout = ',args.dropout)

        # 加载模型
        EastModel = BehaviorModelAutoEncoder(args.zdim , args.dropout)
        SouthEastModel = BehaviorModelAutoEncoder(args.zdim , args.dropout)
        theta1 = torch.Tensor([1])
        theta2 = torch.Tensor([0.1])
        theta3 = torch.Tensor([10])

        # 模型迁移到GPU
        EastModel.to(device)
        SouthEastModel.to(device)
        theta1 = theta1.cuda(device = device)
        theta2 = theta2.cuda(device = device)
        theta3 = theta3.cuda(device = device)
        
        criterion = nn.MSELoss()
        # optimizer = optim.SGD(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001,momentum=0.9)
        # optimizer = optim.Adam(itertools.chain(EastModel.parameters(),SouthEastModel.parameters()),lr = 0.001)
        optimizer = optim.Adam([{'params':EastModel.parameters()},{'params':SouthEastModel.parameters()},{'params':theta1,'lr':0.01},{'params':theta2,'lr':0.01},{'params':theta3,'lr':0.01}],lr = 0.001)
        
        theta1.requires_grad = True
        theta2.requires_grad = True
        theta3.requires_grad = True


        print('Start training.',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) )
        start_time = time.time()
        lastTestingLoss = np.inf
        minPredictionLoss = np.inf

        # 2000个epoch
        for epoch in range(2000):

            running_loss = running_loss1 = running_loss2 = running_loss3 = 0
            count = 0
            EastModel.train()
            SouthEastModel.train()

            lossList = []

            # 训练
            for i in range(9):
                if fakeSingleTrainsets[i].__len__()>0:
                    fakeSingleTrainLoader = DataLoader(fakeSingleTrainsets[i],batch_size=4,shuffle=True)
                else:
                    continue
                count = 0
                for i,sample in enumerate(fakeSingleTrainLoader):
                    trainingPercent = int(100 * (i+1)/fakeSingleTrainLoader.__len__())
                    count += 1
                    E,SE,deltaT = sample['EStateMap'].to(device), sample['SEStateMap'].to(device),sample['deltaT'].to(device)
                    deltaT = deltaT.view(-1,1)
                    optimizer.zero_grad()

                    EOut,Ez = EastModel(E,deltaT)
                    SOut,Sz = SouthEastModel(SE,-deltaT)

                    loss1 = criterion(EOut,SE)
                    loss2 = criterion(SOut,E)
                    loss = (loss1 + loss2)/2

                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    running_loss1 += loss1.item()
                    running_loss2 += loss2.item()

                    lossList.append(loss.item())

                    if count == 1:
                        if fakeSingleTrainLoader.__len__() - (i+1) < count:
                            trainingPercent = 100

                        deltaT = deltaT[0].item()
                        print('[%d, %5d%%]deltaT = %d, training loss: %.3f, E-SE prediction error: %.3f, S-E prediction error: %.3f' %(epoch + 1, trainingPercent,deltaT,running_loss / count,running_loss1/count,running_loss2/count))
                        count = 0
                        running_loss = running_loss1 = running_loss2 = running_loss3 = 0



            running_loss = np.average(lossList)
            tensorboardWriter.add_scalar('training/loss',running_loss,epoch)


            predictionLoss = testing_loss = testing_loss1 = testing_loss2 = testing_loss3 = 0
            count = 0

            EastModel.eval()
            SouthEastModel.eval()
            # 计算当前epoch的testing loss，并可视化部分testing结果
            for i,sample in enumerate(fakeSingleTestLoader):
                E,SE,deltaT = sample['EStateMap'].to(device), sample['SEStateMap'].to(device),sample['deltaT'].to(device)
                deltaT = deltaT.view(-1,1)
                optimizer.zero_grad()

                EOut,Ez = EastModel(E,deltaT)
                SOut,Sz = SouthEastModel(SE,deltaT)

                loss1 = criterion(EOut,SE)
                loss2 = criterion(SOut,E)
                loss = (loss1 + loss2)/2

                predictionLoss += loss.item()


                testing_loss  += loss.item()
                testing_loss1 += loss1.item()
                testing_loss2 += loss2.item()
                count += 1

                if i == 0:
                    concatenate = torch.cat([E,SE,SOut,EOut],0)
                    concatenate = concatenate.detach()
                    concatenate = concatenate.cpu()
                    concatenate = convertDataToBGR(concatenate)
                    concatenate = torchvision.utils.make_grid(concatenate,nrow=4,normalize=False,pad_value=0)

                    concatenate = concatenate.numpy()
                    concatenate = np.transpose(concatenate,(1,2,0))
                    imgName = 'Test_Epoch%d.jpg'%epoch
                    imgName = os.path.join(imgFolder,imgName)
                    cv2.imwrite(imgName,concatenate)

            
            # 保存有史以来predictionLoss最小的网络参数
            if predictionLoss < minPredictionLoss:
                minPredictionLoss = predictionLoss
                torch.save(EastModel.state_dict(),os.path.join(modelParamFolder,'Easemodel.pth'))
                torch.save(SouthEastModel.state_dict(),os.path.join(modelParamFolder,'SEmodel.pth'))


            tensorboardWriter.add_scalar('testing/loss',testing_loss / count,epoch)

            print()
            print('[%d，%6s] testing  loss: %.3f, prediction loss: %.3f, E-SE prediction error: %.3f, S-E prediction error: %.3f' %(epoch + 1,'--', testing_loss / count,predictionLoss/count,testing_loss1/count,testing_loss2/count))

            print()
            print('='*20,end = ' ')
            print('Time is ',time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) ,end = ' ')
            using_time = time.time()-start_time
            hours = int(using_time/3600)
            using_time -= hours*3600
            minutes = int(using_time/60)
            using_time -= minutes*60
            print('running %d h,%d m,%d s'%(hours,minutes,int(using_time)),end = ' ')
            print('='*20)
            print()

    if TestOrTrain == 'test':

        modelParamFolder = '/home/hsc/Research/StateMapPrediction/code/models/EndToEndModel/resultDir/20191211_13_27_24/modelParam'
        typicalTestDataset = typicalTestData(E_dataset_path,SE_dataset_path)
        typicalTestDataLoader = DataLoader(typicalTestDataset,batch_size=4,shuffle=False)

        EastModel = BehaviorModelAutoEncoder()
        SouthEastModel = BehaviorModelAutoEncoder()

        EastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'Easemodel.pth')))
        SouthEastModel.load_state_dict(torch.load(os.path.join(modelParamFolder,'SEmodel.pth')))
        EastModel.to(device)
        SouthEastModel.to(device)
        criterion = nn.MSELoss()


        for i,sample in enumerate(typicalTestDataLoader):
            E,SE,deltaT = sample['EStateMap'].to(device), sample['SEStateMap'].to(device),sample['deltaT'].to(device)
            deltaT = deltaT.view(-1,1)
            SEprediction,Ez = EastModel(E,deltaT)
            Eprediction,Sz = SouthEastModel(SE,deltaT)

            concatenate = torch.cat([E,SE,Eprediction,SEprediction],0)
            concatenate = concatenate.detach()
            concatenate = concatenate.cpu()
            concatenate = convertDataToBGR(concatenate)
            concatenate = torchvision.utils.make_grid(concatenate,nrow=8,normalize=False,pad_value=255)

            concatenate = concatenate.numpy()
            concatenate = np.transpose(concatenate,(1,2,0))
            imgName = '/home/hsc/typicalTestResult.jpg'
            cv2.imwrite(imgName,concatenate)
            print('write img to ', imgName)

            loss = criterion(Ez,Sz)

            loss = (criterion(SEprediction,SE) + criterion(Eprediction,E))/2
            print(loss.item())


            Ez = Ez.detach()
            Ez = Ez.cpu()
            Ez = Ez.numpy()
            Sz = Sz.detach()
            Sz = Sz.cpu()
            Sz = Sz.numpy()




            

        

        
