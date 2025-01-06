# Modules to import for data preprocessing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import random as rnd
import numpy as np

# Modules to import for Torch
import torch
import torch.nn as nn
import torch.optim as optim

# Loading the data into the pandas dataframe
# Reading the Data
data = pd.read_csv("Design_Data.csv", header=0)

# Displaying the structure of the Data 
print(pd.DataFrame(data))

# Inserting the data columns 
data.columns = [
    'Flapping Frequency', 'Airspeed', 'Angle Of Attack', 'Normalised Time',
    'Lift', 'Induced Drag', 'Pitching Moment', 'main wing root chord',
    'main wing wingspan', 'main wing tip chord', 'tail backward position',
    'tail root chord', 'tail tip chord', 'tail wingspan'
]

# Split the data into features and targets
X = data[[
    'Flapping Frequency', 'Airspeed', 'Angle Of Attack', 'Normalised Time',
    'main wing root chord', 'main wing wingspan', 'main wing tip chord',
    'tail backward position', 'tail root chord', 'tail tip chord', 'tail wingspan'
]]
y = data[['Lift', 'Induced Drag', 'Pitching Moment']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to NumPy arrays and then to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Create TensorDatasets for the actual data
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

# Create DataLoaders for training and testing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Printing the Targets and Features
print("Input Parameters :")
print(pd.DataFrame(X_train.cpu().numpy()))
print("\n")
print("Labels :")
print(pd.DataFrame(y_train.cpu().numpy()))

# Check for available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



# Define the neural network model
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Create a model instance and move it to the device
model = DeepNN().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader)}')

print("Training complete.")



# Set the model to evaluation mode
model.eval()

# Initialize lists to store predictions and true values
all_predictions = []
all_targets = []

# Make predictions and evaluate the model
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        all_predictions.extend(outputs.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Convert lists to NumPy arrays
all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)

# Calculate evaluation metrics
mse = mean_squared_error(all_targets, all_predictions)
r2 = r2_score(all_targets, all_predictions)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2) Score: {r2}")
