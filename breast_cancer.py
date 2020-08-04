from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
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

scaler = StandardScaler()

x, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1,1)

x_train, x_val, y_train, y_val = train_test_split(x, y)

x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

initializer = WeightInitializer('he')
layers = [
    Linear(30,64),
    Relu(),
    Linear(64,128),
    Relu(),
    Linear(128,1),
    Sigmoid()
]

model = Model(layers)
criterion = BinaryCrossEntropyLoss()
optimizer = GradientDescent(model.parameters(), learning_rate=0.1)
dataloader = DataLoader(x_train, y_train)

if __name__=='__main__':
    train(dataloader, model, criterion, optimizer, n_epochs=100)
    print()
    print('Train accuracy : ', ((model.forward(x_train) >= 0.5 ) == y_train).mean())
    print('Val accuracy : ', ((model.forward(x_val) >= 0.5) == y_val).mean())
