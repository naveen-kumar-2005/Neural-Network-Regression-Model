# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

It consists of an input layer with 1 neuron, two hidden layers with 8 neurons in the first layer and 10 neurons in the second layer, and an output layer with 1 neuron. Each neuron in one layer is connected to all neurons in the next layer, allowing the model to learn complex patterns. The hidden layers use activation functions such as ReLU to introduce non-linearity, enabling the network to capture intricate relationships within the data. 
During training, the model adjusts its weights and biases using optimization techniques like RMSprop or Adam, minimizing a loss function such as Mean Squared Error for regression.The forward propagation process involves computing weighted sums, applying activation functions, and passing the transformed data through layer.

## Neural Network Model

<img width="1087" height="641" alt="{825AF32E-5D96-4A87-BB84-B0F8AEFC697A}" src="https://github.com/user-attachments/assets/a0896609-69a0-45ab-b0d3-ff436553a9e0" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Naveen Kumar.R
### Register Number: 212223230139
```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```
```python
lig = NeuralNet ()
criterion = nn. MSELoss ()
optimizer = optim.RMSprop (lig. parameters(), lr=0.001)
```

```python
def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=4000) :
  for epoch in range (epochs) :
    optimizer. zero_grad()
    loss = criterion(ai_brain(X_train), y_train)
    loss. backward()
    optimizer.step()
    lig. history['loss'] .append(loss.item())
    if epoch % 200 == 0:
      print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

## Dataset Information

<img width="308" height="400" alt="{5EB3AF21-7742-4514-8E3A-25245A342A18}" src="https://github.com/user-attachments/assets/e6d40b73-98c9-4c43-b3d3-13693368b434" />


## OUTPUT

### Training Loss Vs Iteration Plot

<img width="642" height="483" alt="{BAAC6D68-C2A0-44EB-94CA-14D4D5A24BCA}" src="https://github.com/user-attachments/assets/faea4dc3-0c54-4cd5-84f2-2ac67bb2b398" />


### New Sample Data Prediction
<img width="688" height="131" alt="{E09323DB-5E71-441E-994A-C3C68D97D44B}" src="https://github.com/user-attachments/assets/44598960-59c9-49f3-864c-2eb54da6345f" />

## RESULT

Thus a neural network regression model is developed successfully.The model demonstrated strong predictive performance on unseen data, with a low error rate.
