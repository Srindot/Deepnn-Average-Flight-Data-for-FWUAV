# modules to import for preprocessign for data 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import random as rnd





# modules to import for Torch
import torch
import torch.nn as nn
import torch.optim as optim


# Loading the data into the pandas dataframe

# Reading the Data
data = pd.read_csv("Design_Data.csv",header = 0)

# Displaying the structure of the Data 
print(pd.DataFrame(data))

# Inserting the data columns 
data.columns = ['Flapping Frequency', 'Airspeed', 'Angle Of Attack', 
                'Normalised Time', 'Lift', 'Induced Drag', 'Pitching Moment',"main wing root chord", "main wing wingspan", "main wing tip chord", "tail backward position", "tail root chord", "tail tip chord", "tail wingspan"]



print(pd.DataFrame(data))

# Split the data into features and targets
X = data[[
    'Flapping Frequency', 'Airspeed', 'Angle Of Attack', 'Normalised Time',
    'main wing root chord', 'main wing wingspan', 'main wing tip chord',
    'tail backward position', 'tail root chord', 'tail tip chord', 'tail wingspan'
]]
y = data[['Lift', 'Induced Drag', 'Pitching Moment']]


# Printing the Targets and Features
print("Input Parameters : ")
print(pd.DataFrame(X))
print("\n")
print("Lebels : ")
print(pd.DataFrame(y))



# Defining the MLP Model

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

# Simulate some training data
# Replace this with your actual data loader
input_data = torch.randn(92000, 14).to(device)
output_data = torch.randn(92000, 3).to(device)
dataset = torch.utils.data.TensorDataset(input_data, output_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader)}')

print("Training complete.")



# Set the model to evaluation mode
model.eval()

# Example input data for prediction (replace with your actual data)
new_data = torch.tensor([[0.1, 1.0, -20.0, 0.0, 0.3, 1.4, 0.2, 0.45, 0.2, 0.01, 0.4, 0.3, 1.4, 0.2]], dtype=torch.float32).to(device)

# Set the model to evaluation mode
model.eval()

# Example input data for prediction (replace with your actual data)
new_data = torch.tensor([[0.1, 1.0, -20.0, 0.0, 0.3, 1.4, 0.2, 0.45, 0.2, 0.01, 0.4, 0.3, 1.4, 0.2]], dtype=torch.float32).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(new_data)
    
    # Send predictions to CPU and convert to NumPy array
    predictions_np = predictions.cpu().numpy()

print("Predictions (NumPy array):", predictions_np)


