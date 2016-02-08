import numpy as np
import random
import math
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import datetime as dt

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

codeName = ['BNK금융지주',
            'CJ',
            'CJ대한통운',
            'CJ오쇼핑',
            'CJ제일제당',
            'DGB금융지주',
            'GKL',
            'GS',
            'GS리테일',
            'GS홈쇼핑',
            'KB금융',
            'KB손해보험',
            'KCC',
            'KT&G',
            'LG',
            'LG디스플레이',
            'LG생활건강',
            'LG유플러스',
            'LG전자',
            'LG화학',
            'NAVER',
            'POSCO',
            'S-Oil',
            'SK',
            'SK이노베이션',
            'SK텔레콤',
            'SK하이닉스',
            '강원랜드',
            '고려아연',
            '금호석유',
            '기아차',
            '기업은행',
            '녹십자',
            '다음카카오',
            '대우인터내셔널',
            '대우조선해양',
            '대우증권',
            '동부화재',
            '동서',
            '두산',
            '두산중공업',
            '롯데쇼핑',
            '롯데칠성',
            '롯데케미칼',
            '메디톡스',
            '미래에셋증권',
            '삼성SDI',
            '삼성물산',
            '삼성생명',
            '삼성에스디에스',
            '삼성전기',
            '삼성전자',
            '삼성중공업',
            '삼성증권',
            '삼성카드',
            '삼성화재',
            '셀트리온',
            '신세계',
            '신한지주',
            '아모레G',
            '아모레퍼시픽',
            '아이에스동서',
            '에스에프에이',
            '에스원',
            '엔씨소프트',
            '영원무역',
            '오리온',
            '오스템임플란트',
            '우리은행',
            '유한양행',
            '이마트',
            '이오테크닉스',
            '제일기획',
            '컴투스',
            '코웨이',
            '파라다이스',
            '하나금융지주',
            '하나투어',
            '한국가스공사',
            '한국금융지주',
            '한국전력',
            '한국타이어',
            '한국항공우주',
            '한미약품',
            '한샘',
            '한세실업',
            '한온시스템',
            '한전KPS',
            '한전기술',
            '현대건설',
            '현대그린푸드',
            '현대글로비스',
            '현대리바트',
            '현대모비스',
            '현대백화점',
            '현대위아',
            '현대제철',
            '현대차',
            '현대해상',
            '호텔신라']
out = {};

dCon = 10 # 몇일 전까지 고려하는지
hidden_dim1 = 100 # 히든 레이어의 노드 개수
hidden_dim2 = 50 # 히든 레이어의 노드 개수
featureOfDay = 5 # open high low close volume

mean = [0.0 for x in range(featureOfDay)]
sigma = [0.0 for x in range(featureOfDay)]

fOut = open('0. todaysLuck - All.txt','w')
fOut1 = open('1. todaysLuck - Highly recommend.txt','w')
fOut2 = open('2. todaysLuck - Recommend.txt','w')
fOut3 = open('3. todaysLuck - Risky recommend.txt','w')

for i in range(len(code)): 

    print(str(i+1) ' : ' + str(len(code)) + ' > ' + codeName[i])
    
    try :
        synapse_0 = 2*np.random.random((featureOfDay*dCon, hidden_dim1)) - 1
        synapse_1 = 2*np.random.random((hidden_dim1,hidden_dim2)) - 1
        synapse_2 = 2*np.random.random((hidden_dim2,1)) - 1

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
    except Exception:
        print('파일 불러오기 실패 다음 항목으로')
        continue

        #                                -----------------------------------               read files

    dates = dt.datetime.today();
    dateAgo4 = str(dates.month) + '/' + str(dates.day) + '/'+ str(dates.year)

    out = web.DataReader('KRX:'+code[i], data_source='google', start='09/21/2015', pause = 0) 

    #print(out)
    
    num_col = out.count()
    num = num_col[0]
    kkk = out.keys()
    inputToday = [[0.0 for x in range(featureOfDay * dCon)]]
    count = 0;

    if num != dCon:
        print('입력 자료가 불안정 합니다 해당 종목의 최근 근황을 살펴 보십쇼')
        continue # 안전 장치

    # arrange the data and apply normalization with mean and sigma
    for j in range(num):
        for k in range(featureOfDay): 
            inputToday[0][count] = (out[kkk[k]][j] - mean[k]) / sigma[k]
            count = count+1

    X = inputToday
    layer_1 = 1/(1+np.exp(-(np.dot(X,synapse_0))))
    layer_2 = 1/(1+np.exp(-(np.dot(layer_1,synapse_1))))

    #'''
    if layer_2[[0]] > 0.95:
        print("강력 추천 : ",codeName[i])
        fOut1.write(str(code[i]) + ' ' + codeName[i] + '\n')
    elif layer_2[[0]] > 0.8:
        print("추천 :                  ",codeName[i])
        fOut2.write(str(code[i]) + ' ' + codeName[i] + '\n')
    elif layer_2[[0]] > 0.5:
        print("미흡한 추천                           ",codeName[i])
        fOut3.write(str(code[i]) + ' ' + codeName[i] + '\n')
    else:
        print(code[i], layer_2, codeName[i], out['Close'][3])
    #'''
    
    fOut.write(str(code[i]) + ' ' + codeName[i] + ' ' + str(out['Close'][3]) + ' ' + str(layer_2) + '\n')