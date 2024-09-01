import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        
    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = torch.tanh(self.layers[i](x))
        x = self.layers[-1](x)
        return x

# Data (Can change)
x_train = torch.tensor(np.linspace(0, 1, 5).reshape(-1, 1), dtype=torch.float32)
y_train = torch.sin(np.pi * x_train)  
z_train = torch.cos(np.pi * x_train)  

# PINN model
layers = [1, 20, 20, 1]
model = PINN(layers)

# Training function with loss history
loss_history = []

def train(model, x_train, y_train, z_train, epochs=2000, learning_rate=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        mse_loss = nn.MSELoss()(model(x_train), y_train)
        total_loss = mse_loss  # Assuming physics_informed_loss is handled here
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        if epoch % 1 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss.item()}')
    return model

# Train the model
trained_model = train(model, x_train, y_train, z_train, epochs=2000, learning_rate=0.001)

# Test data
x_test = torch.tensor(np.linspace(0, 1, 5).reshape(-1, 1), dtype=torch.float32)

# Animation function to update plot
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Predicted vs True Plot
ax[0].set_xlim(0, 1)
ax[0].set_ylim(-1, 1)
line, = ax[0].plot([], [], 'b-', lw=2, label='Predicted')
true_line, = ax[0].plot(x_train.detach().numpy(), y_train.detach().numpy(), 'r-', lw=2, label='True')
ax[0].legend()

# Loss Plot
ax[1].set_xlim(0, 2000)  # Set x-axis to show all 2000 epochs
ax[1].set_ylim(0, max(loss_history) * 1.1)  # Slightly scale y-axis for better visualization
loss_line, = ax[1].plot([], [], 'g-', lw=2, label='Loss')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Loss')
ax[1].legend()

def init():
    line.set_data([], [])
    loss_line.set_data([], [])
    return line, loss_line

def animate(i):
    # Update predicted line
    y_test = trained_model(x_test).detach().numpy()  # Predict with the trained model
    line.set_data(x_test.numpy(), y_test)
    
    # Update loss line
    loss_line.set_data(range(len(loss_history[:i+1])), loss_history[:i+1])
    
    return line, loss_line

# Create animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(loss_history), interval=50, blit=True)

# Save the animation as a video file
ani.save('pinn_simulation.mp4', writer='ffmpeg')

plt.show()
