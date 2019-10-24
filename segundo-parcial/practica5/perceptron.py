import numpy as np


class Perceptron(object):
    def __init__(self, numberAttributes, iterations=100, learningRate=0.01):
        self.iterations=iterations
        self.learningRate=learningRate
        self.numberAttributes=numberAttributes
        self.w=np.zeros(numberAttributes+1)

    def predict(self, objectAttributes):
        v = np.dot(objectAttributes, self.w[1:]) + self.w[0] #vector X * W
        y = np.where(v>=0,1,0)
        return y #return the value of y according equation 3 and 4

    def train(self, objetosentrenamiento, etiquetas):
        for x in range(self.iterations): #function to train the perceptron
            for obj, yt in zip(objetosentrenamiento, etiquetas):
                y = self.predict(obj)
                self.w[1:]+=self.learningRate * (yt-y) * obj
                self.w[0] +=  self.learningRate * (yt-y) # x0 = 1
                print(x, self.w, y, yt)

if __name__ == "__main__":
    X=[]
    '''
    AND
    
    X.append(np.array([0, 1]))
    X.append(np.array([1, 0]))
    X.append(np.array([1, 1]))
    X.append(np.array([0, 0]))

    y=np.array([0,0,1,0])
    '''
    print('OR')
    #OR
    X.append(np.array([0, 1]))
    X.append(np.array([1, 0]))
    X.append(np.array([1, 1]))
    X.append(np.array([0, 0]))

    y=np.array([1, 1, 1 ,0])

    perc = Perceptron(2)
    perc.train(X,y)
    print(perc.w)
    print('\n')

    X=[]
    print('XOR')
    # XOR
    X.append(np.array([0, 1]))
    X.append(np.array([1, 0]))
    X.append(np.array([1, 1]))
    X.append(np.array([0, 0]))

    y = np.array([0, 0, 1, 1])

    perc = Perceptron(2)
    perc.train(X, y)

    # NUNCA CONVERGE
    print(perc.w)