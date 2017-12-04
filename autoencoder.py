import _pickle as pickle
import numpy as np 
import torch 
from torch.autograd import Variable
import torch.nn as nn 
import torch.optim as optim 
from random import shuffle
from PIL import Image

def processPred(pred): 

	val = pred.data.numpy().reshape(-1)
	for p in val: 
		if p > 0.7: p = 1
		elif p < 0.3: p = 0
		else: p = 0.5

	val = val.reshape(1,-1)
	tensor = Variable(torch.from_numpy(val).type(torch.FloatTensor))
	return tensor 

def SetCheckpoint(model, name = 'auto.pt'): 

	torch.save(model, name)
	print('Model saved')
# ------------- GET DATA --------------

data = pickle.load(open('data.mehdi', 'rb'))
inp = data[0]
out = data[1]

shuffle(inp)
shuffle(out)

dataSize = pickle.load(open('taille.mehdi', 'rb'))
taille = dataSize[0]
nbDataPoints = dataSize[1]

testPercent = 0.1
nbTestPoint = int(testPercent*nbDataPoints)

trainingIn = inp[0:nbDataPoints - nbTestPoint]
trainingOut = out[0:nbDataPoints - nbTestPoint]

testIn= inp[nbDataPoints - nbTestPoint:nbDataPoints]
testOut = out[nbDataPoints - nbTestPoint:nbDataPoints]


# ----------------- MODEL -----------------

i = taille[0]*taille[1]
h = i
h2 = int(h/2)
o = i

resume = True
if resume: 
	print('Loading model')
	model = torch.load('auto.pt')
	print('Model ready')
else: 

	model = nn.Sequential(nn.Linear(i,h), nn.Sigmoid(), nn.Linear(h,o), nn.Sigmoid())
	#model = nn.Sequential(nn.Linear(i,h), nn.Sigmoid(), nn.Linear(h,h2), nn.Sigmoid(), nn.Linear(h2,o), nn.Sigmoid())
	#model = nn.Sequential(nn.Linear(i,2*h), nn.ReLU(), nn.Linear(h*2, h), nn.ReLU(), nn.Linear(h, int(h/2)), nn.ReLU(), \
	#nn.Linear(int(h/2), h), nn.ReLU(), nn.Linear(h,2*h), nn.ReLU(), nn.Linear(2*h,o), nn.ReLU())

lr = 1e-5
adam = optim.Adam(model.parameters(), lr)

loss_fn = nn.MSELoss()

train = True
# ------------ LEARNING PARAM -------------

epochs = 100
batch_size = 64
iterations = 2000


dtype = torch.FloatTensor

if train: 

	for epoch in range(epochs): 

		for it in range(iterations): 
			indexes = np.random.randint(nbDataPoints - nbTestPoint, size = (batch_size)).tolist()

			x = np.zeros((batch_size,i))
			y = np.zeros((batch_size,o))
			for enum,index in enumerate(indexes):
				x[enum,:] = trainingIn[enum]
				y[enum,:] = trainingOut[enum]

			dx = Variable(torch.from_numpy(x).type(dtype))
			dy = Variable(torch.from_numpy(y).type(dtype))

			pred = model.forward(dx)
			#pred = processPred(pred)
			loss = loss_fn(pred,dy)

			adam.zero_grad()
			loss.backward()
			adam.step()

			if it%(iterations/10) == 0: 
				print ('\n')
				print ('-*-*-*-*-*-*-*-*-*-*-*-*-*-*')
				print ('Epoch: ' + str(epoch) + '/' + str(epochs) + ' | Iterations: ' + str(it) + '/' + str(iterations) + ' | Loss: ' + str(loss))
				print ('-*-*-*-*-*-*-*-*-*-*-*-*-*-*')


		SetCheckpoint(model)


# ------- TIME TO TEST -----------------

print('Starting test')

def postTraite(image, res = [600,600]):

	rx = int(res[0]/taille[0])
	ry = int(res[1]/taille[1])

	nouvelle = np.zeros((res[0], res[1]))
	for h in range(taille[0]): 
		for w in range(taille[1]):
			nouvelle[h*rx:(h+1)*rx, w*ry:(w+1)*ry] = image[h,w]
	return nouvelle

for test in range(nbTestPoint): 

	testX = testIn[test].reshape(1,-1)
	dx = Variable(torch.from_numpy(testX).type(torch.FloatTensor))

	vraie = testOut[test].reshape(1,-1)
	dvraie = Variable(torch.from_numpy(vraie).type(torch.FloatTensor))
	pred = model.forward(dx)

	resultat = pred.data.numpy().reshape(taille[0], taille[1])
	ready = postTraite(resultat)

	path = '/home/mehdi/Codes/Python/GMM5/TestResult/'
	pathP = path + str(test) + 'P.png'
	pathV = path + str(test) + 'V.png'

	dataImage = np.array(ready.T*250, dtype = np.uint8)
	image = Image.fromarray(dataImage,'L')
	image.save(pathP)

	vraie = vraie.reshape(taille[0], taille[1])
	ready = postTraite(vraie)

	dataImage = np.array(ready.T*250, dtype = np.uint8)
	image = Image.fromarray(dataImage,'L')
	image.save(pathV)

print ('Done')

