#!/usr/bin/env python
import numpy as np
#import esig.tosig as ts
from sklearn import linear_model
import sys

#data files
file=['data/nino34.long.data',  'data/amon.us.long.data', \
      'data/np.long.data',      'data/nino12.long.data',  \
      'data/dmi.had.long.data', 'data/soi.long.data',     \
      'data/nao.long.data',     \
      'data/nino3.long.data',  'data/nino4.long.data',\
      'data/tpi.long.data',     'data/ao.long.data']

#parameters
ys  = 1900 #start year including that year
ye  = 2021 #end year including that year
dt  = 1.0/12.0
a   =    6 #lead time
b   =    6 #path length
ord =    3 #order of signature


dy = ye-ys+1
dm = dy*12
nitem=len(file)
dim=nitem+1
#dimension of stream
data2 = np.full((dm,dim),np.nan)
missing = np.zeros(nitem)
for item in range(nitem):
  data = open(file[item])
  i=0; m0=0; i2=0
  for line in data:
    if i==0:
      yse = np.array(line.split(),dtype=int)
    elif i2<=yse[1]-yse[0]:
      yr = np.int(np.array(line.split(),dtype=float)[0])
      if yr>=ys and yr<=ye:
        data2[m0:m0+12,item+1] = np.array(line.split(),dtype=float)[1:13]
        for m in range(m0,m0+12):
          data2[m,0]=yr+(m-m0+0.5)/12.0       #time=y+mon/12.
        m0+=12
      i2+=1
    elif i2==yse[1]-yse[0]+1:
      missing[item]=np.float(line)
      i2+=1
    i+=1
for item in range(nitem):
  for m in range(dm):
    if data2[m,item+1]==missing[item]:
      data2[m,item+1]=np.nan
for item in range(nitem):
  tmp=data2[:,item+1]
  mm = np.nanmean(tmp)
  data2[np.isnan(tmp),item+1] = mm
   
#for m in range(dm):
#  print(*data2[m,:])   
Item=1 #NINO34
#data split
dm2 = dm-a-b+1
y=np.zeros(dm2)
#d=ts.sigdim(dim,ord)
d=b*dim
x=np.zeros((dm2,d))
date=np.zeros(dm2)
vals=np.zeros((dm2,dim))
strm=np.zeros( (b,dim)) # augument 0 at the start
#  print("strm",strm.shape)
m2=0
for m in range(dm2):
    y[m]=data2[m+a+b-1,Item]-data2[m+b-1,Item]    #future vector
    vals[m,:]=data2[m+b-1,:]                  #values at starting date
#    strm[1:,:]=leadlag1(data2[m:m+b,:])                      #past stream
    strm[:,:]=data2[m:m+b,:]                      #past stream
#    strm[0,0]=data2[m,0]                      #past stream
    date[m]=data2[m+a+b-1,0]                  #end date of prediction
#    x[m,:] = ts.stream2sig(strm, ord)         #signature
    x[m,:] = strm.ravel()
#      print(m,strm.shape,x.shape)
    m2+=1
np.savez_compressed('np_savez_comp', x,y,vals,date)
npz_comp = np.load('np_savez_comp.npz')
x=npz_comp['arr_0'];y=npz_comp['arr_1']
vals=npz_comp['arr_2'];date=npz_comp['arr_3']
reg = linear_model.LinearRegression(copy_X=True,fit_intercept=False)


#reg = linear_model.LassoLars(alpha=1,max_iter=10000000,\
#                             copy_X=True,fit_intercept=False,\
#                             normalize=True)

for y0 in range(1979,2022):
 if y0==2021:
   m1 = 2
 else:
   m1 = 12
 for m0 in range(0,m1):
  t0 = y0+m0/12.0
  ofile="ts_ord"+str(ord)+"_"+str(y0)+str(m0+1).zfill(2)+".txt"
  sys.stdout = open(ofile, "w")
 #linear regression
  x_train    = x[date<t0,:]
  y_train    = y[date<t0]
  date_train = date[date<t0]
  vals_train = vals[date<t0,:]
  x_test     = x[date>=t0,:]
  y_test     = y[date>=t0]
  date_test  = date[date>=t0]
  vals_test  = vals[date>=t0,:]

  x_test     = x_test[:a,:]
  y_test     = y_test[:a]
  date_test  = date_test[:a]
  vals_test  = vals_test[:a,:]

  reg.fit(x_train,y_train)
  #NINO34 prediction
  yp = reg.predict(x_test)
#  if (y0==2020 and m0==11) or (y0==2020 and m0==12) or (y0==2021 and m0==1):
#    for j in range(len(date_test)-1,len(date_test)):
#      print(date_test[j],"-",yp[j],vals_test[j,Item],1)
#  else:
  for j in range(len(date_test)-1,len(date_test)):
    print(date_test[j],y_test[j],yp[j],vals_test[j,Item],1)
n_nonzero = np.sum(reg.coef_ != 0)
n_zero = np.sum(reg.coef_ == 0)
print("#NONZERO", n_nonzero,n_zero)
#sw: Standard partial regression coefficient (標準偏回帰係数)
#key=ts.sigkeys(dim,ord).split()
#sx = np.zeros(len(x_train[0,:]))
#for i in range(len(x_train[0,:])):
# sx[i] = np.std(x_train[:,i])
#sy    = np.std(y_train[:])
#nino = 0.0; ninod = 0.0
#for i in range(len(reg.coef_)):
# nino += reg.coef_[i]*x_test[-1,i]
# sw = reg.coef_[i]*sx[i]/sy
# if np.abs(sw) > 0.01:
#    ninod += reg.coef_[i]*x_test[-1,i]
#    print("#",i, key[i], sw, np.abs(sw))
#print("#total ",nino, ninod,y_test[-1])
