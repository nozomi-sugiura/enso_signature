#!/usr/bin/env python
def calc_sw(coef,x_train,y_train,ord,dim):
#sw: Standard partial regression coefficiets
  key=ts.sigkeys(dim,ord).split()
  sw = np.zeros(len(coef))
  sx = np.zeros(len(x_train[0,:]))
  sx2 = np.zeros(len(x_train[0,:]))
  for i in range(len(x_train[0,:])):
   sx[i] = np.std(x_train[:,i])
   sx2[i] = np.std(x_train[:,i]*coef[i])
  sy    = np.std(y_train[:])
  for i in range(len(coef)):
   sw[i] = coef[i]*sx[i]/sy
   if np.abs(sw[i]) > 0.01:
     print("#",i, key[i], sw[i], np.abs(sw[i]),sx2[i])
  return sw 
def leadlag1(X):
  dim0 = X.shape[0]  
  dim1 = X.shape[1]
  l = X[0,:]
  ll = np.zeros(((dim0-1)*dim1+1,dim1))
  t=0
  ll[t,:] = l
  for j in range(1,dim0):
      for k in range(dim1):
          t+=1
          l[k] = X[j,k]
          ll[t,:] = l
  return ll
def leadlag(X):
  dim0 = X.shape[0]  
  dim1 = X.shape[1]
  l = np.concatenate((X[0,:],X[0,:]), axis = 0)
  ll = np.zeros(((dim0-1)*2*dim1+1,2*dim1))
  t=0
  ll[t,:] = l
  for j in range(1,dim0):
      for k in range(2*dim1):
          t+=1
          l[k] = X[j,k%2]
          ll[t,:] = l
  return ll
import numpy as np
import esig.tosig as ts
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

#for item in range(nitem):
#  for m in range(dm):
#    if data2[m,item+1]==missing[item]:
#      data2[m,item+1]=data2[m-1,item+1]
#Item=7 #pdo
Item=1 #NINO34
#data split
dm2 = dm-a-b+1
y=np.zeros(dm2)
d=ts.sigdim(dim,ord)
x=np.zeros((dm2,d))
date=np.zeros(dm2)
vals=np.zeros((dm2,dim))
strm=np.zeros( (b+1,dim)) # augument 0 at the start
#  print("strm",strm.shape)
m2=0
for m in range(dm2):
    y[m]=data2[m+a+b-1,Item]-data2[m+b-1,Item]    #future vector
    vals[m,:]=data2[m+b-1,:]                  #values at starting date
    strm[1:,:]=data2[m:m+b,:]                      #past stream
    strm[0,0]=data2[m,0]                      #past stream
    date[m]=data2[m+a+b-1,0]                  #end date of prediction
    x[m,:] = ts.stream2sig(strm, ord)         #signature
#      print(m,strm.shape,x.shape)
    m2+=1
np.savez_compressed('np_savez_comp', x,y,vals,date)
npz_comp = np.load('np_savez_comp.npz')
x=npz_comp['arr_0'];y=npz_comp['arr_1']
vals=npz_comp['arr_2'];date=npz_comp['arr_3']
#reg = linear_model.LinearRegression(copy_X=True,fit_intercept=False)
reg = linear_model.LassoLars(alpha=2,max_iter=10000000,\
                             copy_X=True,fit_intercept=False,\
                             normalize=False)
#1=>0.606
#2=>0.596
#2.5=>0.596
#3=>0.597
#4=>0.600
sws=np.zeros(d)
nws=np.zeros(d)
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
  sc=reg.score(x_train, y_train)
  #NINO34 prediction
  yp = reg.predict(x_test)
#  if (y0==2020 and m0==11) or (y0==2020 and m0==12) or (y0==2021 and m0==1):
#    for j in range(len(date_test)-1,len(date_test)):
#      print(date_test[j],"-",yp[j],vals_test[j,Item],sc)
#  else:
  for j in range(len(date_test)-1,len(date_test)):
    print(date_test[j],y_test[j],yp[j],vals_test[j,Item],sc)
  swt = calc_sw(reg.coef_,x_train,y_train,ord,dim)
  sws += swt
  nws += 1
#sw: Standard partial regression coefficient (標準偏回帰係数)
sws=sws/nws
#key=ts.sigkeys(dim,ord).split()
#for i in range(d):
# sw = sws[i]
# if np.abs(sw) > 0.01:
#    print("#",i, key[i], sw, np.abs(sw))

n_nonzero = np.sum(reg.coef_ != 0)
print("#NONZERO", n_nonzero)
