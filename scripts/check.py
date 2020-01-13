import pickle
with open('data.pickle','rb') as f:
    train,test = pickle.load(f)
cnt = 0
for x in train:
    if x['is entity']=='1':
        cnt+=1
print(cnt,len(train))
cnt = 0
for x in test:
    if x['is entity']=='1':
        cnt+=1
print(cnt,len(test))

