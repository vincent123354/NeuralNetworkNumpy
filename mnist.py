from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src import *

def train(dataloader, model, criterion, optimizer, n_epochs=10):
    for epoch in range(1,n_epochs+1):
        running_loss = 0
        for idx, (x, y) in enumerate(dataloader):
            print('\rEpoch {:d}\t\t{:.2f}%'.format(epoch+1, (idx+1)/len(dataloader)*100), end='')
            predict = model.forward(x)
            loss = criterion(predict, y)
            model.backward(criterion)
            optimizer.step()
            
            running_loss += loss
        
        # if epoch % (n_epochs//10) == 0: 
        print('\rEpoch {:d}\t\tLoss : {:.10f}'.format(epoch+1, running_loss/len(dataloader)))

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

x = x.reshape(-1,1,28,28)
x/=x.max()
y = y.reshape(-1,1).astype(int)
y = Utils.onehot(y)

x_train, x_val, y_train, y_val = train_test_split(x, y)

initializer = WeightInitializer('he')
layers = [
    Conv2d(1,4,3,initializer=initializer),
    AvgPool(3),
    Relu(),
    Conv2d(4,16,3,initializer=initializer),
    AvgPool(3),
    Relu(),
    Flatten(),
    Linear(64,32,initializer=initializer),
    Relu(),
    Linear(32,16,initializer=initializer),
    Relu(),
    Linear(16, 10,initializer=initializer),
    Softmax()
]

model = Model(layers)
criterion = CrossEntropyLoss()
optimizer = GradientDescent(model.parameters(), learning_rate=0.1)
dataloader = DataLoader(x_train, y_train)


if __name__=='__main__':
    train(dataloader, model, criterion, optimizer, n_epochs=10)
    print()
    print('Train accuracy : ', (np.argmax(Softmax()(model.forward(x_train)), axis=1) == np.argmax(y_train, axis=1)).mean())
    print('Val accuracy : ', (np.argmax(Softmax()(model.forward(x_val)), axis=1) == np.argmax(y_val, axis=1)).mean())
