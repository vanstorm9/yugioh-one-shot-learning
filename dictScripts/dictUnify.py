import os 
import pickle

targetDir = './dict/'

resDict = None
for i,files in enumerate(os.listdir(targetDir)):
    absPath = targetDir+files

    if resDict is None:
        resDict = pickle.load( open(absPath, "rb" ))
    else:
        tmpDict = pickle.load( open(absPath, "rb" ))
        resDict.update(tmpDict)

print('Length of new dictionary: ', len(resDict))

savePath = './featureMap-combined.pkl'
f = open(savePath,"wb")
pickle.dump(resDict,f)
f.close()
print('Dictionary saved!')


