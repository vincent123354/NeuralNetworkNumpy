from sklearn.datasets import load_digits
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
        
        if epoch % (n_epochs//10) == 0: 
            print('\rEpoch {:d}\t\tLoss : {:.10f}'.format(epoch+1, running_loss/len(dataloader)))

x, y = load_digits(n_class=10, return_X_y=True)

x = x.reshape(-1,1,8,8)
x /= x.max()
y = y.reshape(-1,1)

y = Utils.onehot(y)

x_train, x_val, y_train, y_val = train_test_split(x, y)

initializer = WeightInitializer('he')
layers = [
    Conv2d(1,4,3,initializer=initializer),
    Relu(),
    Conv2d(4,16,3,initializer=initializer),
    Relu(),

    Flatten(),
    Linear(256,32,initializer=initializer),
    Relu(),
    Linear(32,64,initializer=initializer),
    Relu(),
    Linear(64, 10,initializer=initializer),
    Softmax()
]

model = Model(layers)
criterion = CrossEntropyLoss()
optimizer = GradientDescent(model.parameters(), learning_rate=0.01)
dataloader = DataLoader(x_train, y_train)


if __name__=='__main__':
    train(dataloader, model, criterion, optimizer, n_epochs=30)
    print()
    print('Train accuracy : ', (np.argmax(Softmax()(model.forward(x_train)), axis=1) == np.argmax(y_train, axis=1)).mean())
    print('Val accuracy : ', (np.argmax(Softmax()(model.forward(x_val)), axis=1) == np.argmax(y_val, axis=1)).mean())
