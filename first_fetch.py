import numpy as np
import pandas as pd
import pandas.io.data as web
import matplotlib.pyplot as plt
import math

code = ['138930',	'001040',	'000120',	'035760',	'097950',	'139130',	'114090',	'078930',	'007070',	'028150',	'105560',
    	'002550',	'002380',	'033780',	'003550',	'034220',	'051900',	'032640',	'066570',	'051910',	'035420',	'005490',
    	'010950',	'034730',	'096770',	'017670',	'000660',	'035250',	'010130',	'011780',	'000270',	'024110',	'006280',
    	'035720',	'047050',	'042660',	'006800',	'005830',	'026960',	'000150',	'034020',	'023530',	'005300',	'011170',
    	'086900',	'037620',	'006400',	'028260',	'032830',	'018260',	'009150',	'005930',	'010140',	'016360',	'029780',
    	'000810',	'068270',	'004170',	'055550',	'002790',	'090430',	'010780',	'056190',	'012750',	'036570',	'111770',
    	'001800',	'048260',	'000030',	'000100',	'139480',	'039030',	'030000',	'078340',	'021240',	'034230',	'086790',
    	'039130',	'036460',	'071050',	'015760',	'161390',	'047810',	'128940',	'009240',	'105630',	'018880',	'051600',
    	'052690',	'000720',	'005440',	'086280',	'079430',	'012330',	'069960',	'011210',	'004020',	'005380',	'001450',
    	'008770']
out = {}

featureOfDay = 5
dCon = 10

for i in range(len(code)): 
    try :
        mean = [0.0 for x in range(featureOfDay)]
        sigma = [0.0 for x in range(featureOfDay)]

        out = web.DataReader('KRX:'+code[i], data_source='google', start='01/01/1990', pause = 0.1) 
        # out[i][['Close']].plot(grid=True, figsize=(10,8))

        print("%d / 100 fetching data from google" % (i+1))
        f = open("./fetchData/" + code[i] + '.txt','w')

        num_col = out.count()
        num = num_col[0]
        kkk = out.keys()

        print(str(num) + " to " + str(num-2))
        f.write(str(num-2)+" \n")

        #Date Open High Low Close Volume
        
        for k in range(5):
            for j in range(num):
                mean[k] = mean[k] + (out[kkk[k]][j] / num)
            for i1 in range(num):
                sigma[k] = sigma[k] + ((out[kkk[k]][j]-mean[k])*(out[kkk[k]][j]-mean[k]) / (num-1))
            sigma[k] = math.sqrt(sigma[k])

        for k in range(num-2):

            if k > dCon-2:
                y = (out['Open'][k+2] > out['Open'][k+1] * 1.02)
            else:
                y = -1

            f.write(str(int(y))+" ")
            for j in range(5):
                f.write( str( ( (out[kkk[j]][k]-mean[j]) / sigma[j] ) )+" ")

            f.writelines("\n")

        for j in range(5):
            f.write(str(mean[j])+" ")
        f.writelines("\n")

        for j in range(5):
            f.write(str(sigma[j])+" ")
        f.writelines("\n")

    except Exception:
        #print ( Exception.with_traceback() )
        print ( code[i], " occurs problems" )
