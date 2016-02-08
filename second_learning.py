import numpy as np
import random
import math
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt

'''
X = np.array([ [0,0,1],[0,1,1],[1,0,1],[1,1,1] ])
y = np.array([[0,1,1,0]]).T

print(X)
print(y)

alpha,hidden_dim = (0.2, 3)
synapse_0 = 2*np.random.random((3,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,1)) - 1

for j in range(60000):
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))

    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
    synapse_0 -= (alpha * X.T.dot(layer_1_delta))
    
print(synapse_1)

layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

print(layer_2)
'''

code = ['138930',	'001040',   '000120',	'035760',	'097950',	'139130',	'114090',	'078930',	'007070',	'028150',	'105560',
    	'002550',	'002380',	'033780',	'003550',	'034220',	'051900',	'032640',	'066570',	'051910',	'035420',	'005490',
    	'010950',	'034730',	'096770',	'017670',	'000660',	'035250',	'010130',	'011780',	'000270',	'024110',	'006280',
    	'035720',	'047050',	'042660',	'006800',	'005830',	'026960',	'000150',	'034020',	'023530',	'005300',	'011170',
    	'086900',	'037620',	'006400',	'028260',	'032830',	'018260',	'009150',	'005930',	'010140',	'016360',	'029780',
    	'000810',	'068270',	'004170',	'055550',	'002790',	'090430',	'010780',	'056190',	'012750',	'036570',	'111770',
    	'001800',	'048260',	'000030',	'000100',	'139480',	'039030',	'030000',	'078340',	'021240',	'034230',	'086790',
    	'039130',	'036460',	'071050',	'015760',	'161390',	'047810',	'128940',	'009240',	'105630',	'018880',	'051600',
    	'052690',	'000720',	'005440',	'086280',	'079430',	'012330',	'069960',	'011210',	'004020',	'005380',	'001450',
    	'008770']


dCon = 10 # 몇일 전까지 고려하는지
hidden_dim1 = 1000 # 히든 레이어의 노드 개수
hidden_dim2 = 500 # 히든 레이어의 노드 개수
featureOfDay = 5 # open high low close volume

mean = [0.0 for x in range(featureOfDay)]
sigma = [0.0 for x in range(featureOfDay)]

for i in range(len(code)):     

    count = 0;
    mTrainX = [[0.0 for x in range(featureOfDay)] for x in range(5000)] 
    mTrainY = [[0] for x in range(5000)]
    temp = 0
        
    try :
        f = open('./fetchData/' + code[i] + '.txt','r')
        
        data = f.readline()
        sep_data = str(data).split(sep = ' ')
        sep_data.remove('\n')
        count = int(sep_data[0]) # mTrain 이 몇개? count 개!!!

        for ii in range(count):
            data = f.readline()
            sep_data = str(data).split(sep=' ')
            sep_data.remove('\n')
            mTrainY[ii][0] = int(sep_data[0]) # Y 값

            for kk in range(featureOfDay):
                mTrainX[ii][kk] = float(sep_data[kk+1]) # Feature 값

        data = f.readline()
        sep_data = str(data).split(sep=' ')
        sep_data.remove('\n')
        for kk in range(featureOfDay):
            mean[kk] = float(sep_data[kk]);

        data = f.readline()
        sep_data = str(data).split(sep=' ')
        sep_data.remove('\n')
        for kk in range(featureOfDay):
            sigma[kk] = float(sep_data[kk]);

    except Exception:
        print('getting train Set occur problem' + code[i])
        continue

    updated_count = 0
        
    synapse_0 = 2*np.random.random((featureOfDay*dCon, hidden_dim1)) - 1
    synapse_1 = 2*np.random.random((hidden_dim1,hidden_dim2)) - 1
    synapse_2 = 2*np.random.random((hidden_dim2,1)) - 1
    
    try :
        f = open("./learnedWeight/" + code[i] + '.txt','r')

        data = f.readline()
        sep_data = str(data).split(sep=' ')
        sep_data.remove('\n')
        updated_count = int(sep_data[0]) # 업데이트 된 횟수
            
        data = f.readline() # 평균값
        sep_data = str(data).split(sep=' ')
        sep_data.remove('\n')
        for kk in range(featureOfDay):
            mean[kk] = float(sep_data[kk]);

        data = f.readline() # 표준편차
        sep_data = str(data).split(sep=' ')
        sep_data.remove('\n')
        for kk in range(featureOfDay):
            sigma[kk] = float(sep_data[kk]);

        print('1')

        data = f.readline() # 엔터 소거
        for i1 in range(featureOfDay*dCon):
            data = f.readline() # 시냅스 1
            sep_data = str(data).split(sep=' ')
            sep_data.remove('\n')
            for i2 in range(hidden_dim1):
                synapse_0[i1][i2] = float(sep_data[i2])

        print('1')

        data = f.readline() # 엔터 소거
        for i1 in range(hidden_dim1):
            data = f.readline() # 시냅스 2
            sep_data = str(data).split(sep=' ')
            sep_data.remove('\n')
            for i2 in range(hidden_dim2):
                synapse_1[i1][i2] = float(sep_data[i2])
        
        print('1')
        
        data = f.readline()
        data = f.readline() # 엔터 소거
        sep_data = str(data).split(sep=' ')
        for i1 in range(hidden_dim2): # 시냅스 3
            synapse_2[i1][0] = float(sep_data[i1])
        
        f.close()    
        print('세타 불러오기 완료' + str(code[i]))
            
    except : 
        #print(Exception.with_traceback()) 
        print('신규 작업 합니다' + str(code[i]))
        updated_count = 0    

    # Seperate training set & Validation set

    
    postive_count = 0
    

    noGroupX = [[0 for x in range(featureOfDay * dCon)] for x in range(count-dCon+1)]
    noGroupY = [[0] for x in range(count-dCon+1)]

    for ii in range(dCon-1,count): # 10일당 하나로 붙여~~~
        for put in range(dCon):
            minus = (dCon-1) - put
            for lop in range(featureOfDay):
                #print(ii,put*featureOfDay + lop,ii-minus)
                noGroupX[ii-dCon+1][put*featureOfDay + lop] = mTrainX[ii-minus][lop]
        

        noGroupY[ii-dCon+1][0] = mTrainY[ii][0]

    count = count - dCon + 1 # 전체 데이터 셋 수 조정

    '''
    ff = open('./dataX.txt','w')
    fff = open('./dataY.txt','w')
    ff.write(str(count)+'\n')
    for ii in range(count):
        fff.write(str(noGroupY[ii][0]) + "\n")
        for jj in range(50):
            ff.write(str(noGroupX[ii][jj]) + " ")
        ff.write("\n")
    '''

    print(count)
    nSample = int(count/10)  # 한 세트의 train set
    nValidation = int(count/10) + count%10 # validation set

    trainSetX = [[[0.0 for x in range(featureOfDay * dCon)] for x in range(nSample)] for x in range(9)]
    validSetX =  [[0.0 for x in range(featureOfDay * dCon)] for x in range(nValidation)]
    trainSetY = [[[0] for x in range(nSample)] for x in range(9)]
    validSetY =  [[0] for x in range(nValidation)]

    validList = [False for x in range(5000)]

    flag = [0 for x in range(10)]
    ii = 0

    while (ii != count): # 9개와 1개의 셋으로 나눔
        number = int(random.random()*10)

        if number == 9 : #validation
            if(flag[number] == nValidation):
                continue
            validList[ii] = True;

            validSetX[flag[number]] = noGroupX[ii]
            validSetY[flag[number]][0] = noGroupY[ii][0]
            flag[number] = flag[number]+1
        else : # test set
            if(flag[number] == nSample):
                continue
            trainSetX[number][flag[number]] = noGroupX[ii]
            trainSetY[number][flag[number]][0] = noGroupY[ii][0]
            flag[number] = flag[number]+1
        
        ii = ii + 1

    #print(nSample, nValidation)
    alpha = 0.02
    keep = 1
        
    while keep:
        temp = flag[9]
        flag = [0 for x in range(10)]
        flag[9] = temp
        ii = 0
        keep = 1

        while (ii != count): # 9개와 1개의 셋으로 나눔
            number = int(random.random()*10)
            if(validList[ii]) :
                ii = ii + 1;
                continue
            elif number == 9 :
                continue
            else : # test set
                if(flag[number] == nSample):
                    continue
                trainSetX[number][flag[number]] = noGroupX[ii]
                trainSetY[number][flag[number]][0] = noGroupY[ii][0]
                flag[number] = flag[number]+1
        
            ii = ii + 1
        
            

        print(flag)
        print('seperation done')

        # weight intialization ---------------------------------------------------------------------------------------
        nTryWeight = 0 
        iterationNum = 10;

        nCorrect_11 = 0
        nCorrect_00 = 0
        nWrong_10 = 0
        nWrong_01 = 0 # it's harmful
        
        while True: # find weights    
        
            print ( nTryWeight, alpha)
            nTryWeight = nTryWeight+1
            sumErr = 0
    
            for smSet in range(9):

                X = np.array(trainSetX[smSet])
                y = np.array(trainSetY[smSet]) # 트레이닝 세트 준비 완료

                for j in range(iterationNum):
                    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
                    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
                    layer_3 = 1/(1+np.exp(-(np.dot(layer_2,synapse_2))))

                    layer_3_delta = (layer_3 - y)*(layer_3*(1-layer_3))
                    layer_2_delta = layer_3_delta.dot(synapse_2.T) * (layer_2*(1-layer_2))
                    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
            
                    layer_3_error = y - layer_3
                    '''
                    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
                    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

                    layer_2_delta = (layer_2 - y)*(layer_2*(1-layer_2))
                    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1-layer_1))
                    layer_2_error = y - layer_2
                    '''
                    #print (layer_2_delta)
                    if (j == iterationNum-1) :
                        #print (alpha)
                        #print (layer_1_delta)
                        sumErr = sumErr + np.mean(np.abs(layer_3_error))
        
                    synapse_2 -= (alpha * layer_2.T.dot(layer_3_delta))
                    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
                    synapse_0 -= (alpha * X.T.dot(layer_1_delta))
                    '''
                    synapse_1 -= (alpha * layer_1.T.dot(layer_2_delta))
                    synapse_0 -= (alpha * X.T.dot(layer_1_delta))
                    '''
                    #
                    #if (j % 1000 == 0):
                    #    print(j/1000)
                    #
            # 학습 완료 9 블럭    

            print ("Error:" + str(sumErr))
            #print(layer_2)
            #print('descent done')
        
           # print(layer_2)

            X = np.array(validSetX)
            y = np.array(validSetY) # validation 세트 준비 완료

            layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
            layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))
            layer_3 = 1/(1+np.exp(-(np.dot(layer_2,synapse_2))))
            nCorrect_11 = 0
            nCorrect_00 = 0
            nWrong_10 = 0
            nWrong_01 = 0 # it's harmful

            theshold = 0.5

            for i1 in range(flag[9]):
                if (layer_3[i1][0] <= theshold and y[i1][0] == 0):
                    nCorrect_00 = nCorrect_00+1
                elif (layer_3[i1][0] <= theshold and y[i1][0] == 1):
                    nWrong_10 = nWrong_10 + 1
                elif (layer_3[i1][0] > theshold and y[i1][0] == 1):
                    nCorrect_11 = nCorrect_11 + 1
                else:
                    nWrong_01 = nWrong_01 + 1

           # print(layer_2)

            updated_count = updated_count + 1

            # mean, sigma, weight 기록
            f = open("./learnedWeight/" + code[i] + '.txt','w')
            f.write(str(updated_count) + " ")
            f.write("\n")

            for i1 in range(featureOfDay):
                f.write(str(mean[i1]) + " ")
            f.write("\n")

            for i1 in range(featureOfDay):
                f.write(str(sigma[i1]) + " ")
            f.write("\n")
            f.write("\n")

            for i1 in range(featureOfDay*dCon):
                for i2 in range(hidden_dim1):
                    f.write(str(synapse_0[i1][i2]) + " ")
                f.write("\n")
       
            f.write("\n")
            for i1 in range(hidden_dim1):
                for i2 in range(hidden_dim2):
                    f.write(str(synapse_1[i1][i2]) + " ")
                f.write("\n")
                    
            f.write("\n")
            for i1 in range(hidden_dim2):
                f.write(str(synapse_2[i1][0]) + " ")

            f.close()
            # 트레이닝과 신뢰도 검증 완료
        
            print('1 as 1', nCorrect_11, '0 as 0',nCorrect_00)
            print('1 as 0', nWrong_10, '0 as 1', nWrong_01)
            print(keep, ((nCorrect_00 + nCorrect_11) / flag[9])*100 )
            print ('\n')
            # 현황 출력과, 신뢰도 90% 이상일 경우 종료.
            if (((nCorrect_00+nCorrect_11) / flag[9] > 0.85) and (nCorrect_11 > nWrong_01)):
                keep = 0
            elif nTryWeight > 20: # 오버 핏 되고 결과가 별로 이면 다시 뽑아
                alpha = alpha * 0.99
                break

        if (keep == 0):
            break

    print('motherTrainSet :',count,'current',i, '/', 100, postive_count)

#'''