import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt, pi
import time
import vtk
from vtk.util import numpy_support as VN

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            return x.mul_(torch.sigmoid(x))
        else:
            return x * torch.sigmoid(x)

class Net2_u(nn.Module):
    def __init__(self, input_n=3, h_n=200):
        super(Net2_u, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),
        )

    def forward(self, x):
        return self.main(x)

class Net2_v(nn.Module):
    def __init__(self, input_n=3, h_n=200):
        super(Net2_v, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),
        )

    def forward(self, x):
        return self.main(x)

class Net2_w(nn.Module):
    def __init__(self, input_n=3, h_n=200):
        super(Net2_w, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),
        )

    def forward(self, x):
        return self.main(x)

class Net2_p(nn.Module):
    def __init__(self, input_n=3, h_n=200):
        super(Net2_p, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, h_n),
            Swish(),
            nn.Linear(h_n, 1),
        )

    def forward(self, x):
        return self.main(x)

def geo_train(device, x_in, y_in, z_in, xb, yb, zb, ub, vb, wb, xd, yd, zd, ud, vd, wd,
              batchsize, learning_rate, epochs, path, Flag_batch, Diff, rho, Flag_BC_exact,
              Lambda_BC, nPt, T, xb_inlet, yb_inlet, zb_inlet, ub_inlet, vb_inlet, wb_inlet):
    
    # Convert inputs to tensors
    x = torch.Tensor(x_in).to(device)
    y = torch.Tensor(y_in).to(device)
    z = torch.Tensor(z_in).to(device)
    xb = torch.Tensor(xb).to(device)
    yb = torch.Tensor(yb).to(device)
    zb = torch.Tensor(zb).to(device)
    ub = torch.Tensor(ub).to(device)
    vb = torch.Tensor(vb).to(device)
    wb = torch.Tensor(wb).to(device)
    xd = torch.Tensor(xd).to(device)
    yd = torch.Tensor(yd).to(device)
    zd = torch.Tensor(zd).to(device)
    ud = torch.Tensor(ud).to(device)
    vd = torch.Tensor(vd).to(device)
    wd = torch.Tensor(wd).to(device)
    xb_inlet = torch.Tensor(xb_inlet).to(device)
    yb_inlet = torch.Tensor(yb_inlet).to(device)
    zb_inlet = torch.Tensor(zb_inlet).to(device)
    ub_inlet = torch.Tensor(ub_inlet).to(device)
    vb_inlet = torch.Tensor(vb_inlet).to(device)
    wb_inlet = torch.Tensor(wb_inlet).to(device)
    
    if Flag_batch:
        dataset = TensorDataset(x, y, z)
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=True, drop_last=True)
    else:
        raise NotImplementedError("Non-batch mode is not implemented yet.")

    # Initialize models
    net2_u = Net2_u().to(device)
    net2_v = Net2_v().to(device)
    net2_w = Net2_w().to(device)
    net2_p = Net2_p().to(device)

    # Initialize weights
    def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
    
    net2_u.apply(init_normal)
    net2_v.apply(init_normal)
    net2_w.apply(init_normal)
    net2_p.apply(init_normal)

    # Optimizers
    optimizer_u = optim.Adam(net2_u.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    optimizer_v = optim.Adam(net2_v.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    optimizer_w = optim.Adam(net2_w.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    optimizer_p = optim.Adam(net2_p.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)

    def criterion(x, y, z):
        x.requires_grad = True
        y.requires_grad = True
        z.requires_grad = True

        net_in = torch.cat((x, y, z), 1)
        u = net2_u(net_in)
        v = net2_v(net_in)
        w = net2_w(net_in)
        P = net2_p(net_in)

        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        v_z = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        w_x = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        w_y = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        w_z = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        P_x = torch.autograd.grad(P, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        P_y = torch.autograd.grad(P, y, grad_outputs=torch.ones_like(y), create_graph=True, only_inputs=True)[0]
        P_z = torch.autograd.grad(P, z, grad_outputs=torch.ones_like(z), create_graph=True, only_inputs=True)[0]

        loss_f = nn.MSELoss()

        # Define losses
        loss_2 = u * u_x + v * u_y + w * u_z - Diff * (u_x + u_y + u_z) + P_x / rho
        loss_1 = u * v_x + v * v_y + w * v_z - Diff * (v_x + v_y + v_z) + P_y / rho
        loss_3 = u_x + v_y + w_z  # continuity
        loss_4 = u * w_x + v * w_y + w * w_z - Diff * (w_x + w_y + w_z) + P_z / rho

        loss = loss_f(loss_1, torch.zeros_like(loss_1)) + \
               loss_f(loss_2, torch.zeros_like(loss_2)) + \
               loss_f(loss_3, torch.zeros_like(loss_3)) + \
               loss_f(loss_4, torch.zeros_like(loss_4))

        return loss

    def Loss_BC(xb, yb, zb, ub, vb, wb, xb_inlet, yb_inlet, ub_inlet):
        net_in1 = torch.cat((xb, yb, zb), 1)
        out1_u = net2_u(net_in1)
        out1_v = net2_v(net_in1)
        out1_w = net2_w(net_in1)

        loss_f = nn.MSELoss()
        loss_noslip = loss_f(out1_u, torch.zeros_like(out1_u)) + \
                      loss_f(out1_v, torch.zeros_like(out1_v)) + \
                      loss_f(out1_w, torch.zeros_like(out1_w))
        return loss_noslip

    def Loss_data(xd, yd, zd, ud, vd, wd):
        net_in1 = torch.cat((xd, yd, zd), 1)
        out1_u = net2_u(net_in1)
        out1_v = net2_v(net_in1)
        out1_w = net2_w(net_in1)

        loss_f = nn.MSELoss()
        loss_d = loss_f(out1_u, ud) + loss_f(out1_v, vd) + loss_f(out1_w, wd)
        return loss_d

    tic = time.time()

    if Flag_batch:
        for epoch in range(epochs):
            loss_eqn_tot = 0.
            loss_bc_tot = 0.
            loss_data_tot = 0.
            n = 0
            for batch_idx, (x_in, y_in, z_in) in enumerate(dataloader):
                optimizer_u.zero_grad()
                optimizer_v.zero_grad()
                optimizer_w.zero_grad()
                optimizer_p.zero_grad()

                loss_eqn = criterion(x_in, y_in, z_in)
                loss_bc = Loss_BC(xb, yb, zb, ub, vb, wb, xb_inlet, yb_inlet, ub_inlet)
                loss_data = Loss_data(xd, yd, zd, ud, vd, wd)
                loss = loss_eqn + Lambda_BC * loss_bc + Lambda_data * loss_data
                loss.backward()

                optimizer_u.step()
                optimizer_v.step()
                optimizer_w.step()
                optimizer_p.step()

                loss_eqn_tot += loss_eqn.item()
                loss_bc_tot += loss_bc.item()
                loss_data_tot += loss_data.item()
                n += 1

                if batch_idx % 40 == 0:
                    print(f"Train Epoch: {epoch} [{batch_idx * len(x_in)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]"
                          f"\tLoss eqn: {loss_eqn.item():.10f} Loss BC: {loss_bc.item():.8f} Loss data: {loss_data.item():.8f}")

            loss_eqn_tot /= n
            loss_bc_tot /= n
            loss_data_tot /= n
            print(f"*****Total avg Loss : Loss eqn: {loss_eqn_tot:.10f} Loss BC: {loss_bc_tot:.10f} Loss data: {loss_data_tot:.10f} ****")

            if epoch % 1000 == 0 and epoch > 3000:
                torch.save(net2_p.state_dict(), f"{path}IA3D_data3velsmall_p.pt")
                torch.save(net2_u.state_dict(), f"{path}IA3D_data3velsmall_u.pt")
                torch.save(net2_v.state_dict(), f"{path}IA3D_data3velsmall_v.pt")
                torch.save(net2_w.state_dict(), f"{path}IA3D_data3velsmall_w.pt")

    toc = time.time()
    elapseTime = toc - tic
    print("Elapsed time in parallel = ", elapseTime)

    # Save models at the end
    torch.save(net2_p.state_dict(), f"{path}IA3D_data3velsmall_p.pt")
    torch.save(net2_u.state_dict(), f"{path}IA3D_data3velsmall_u.pt")
    torch.save(net2_v.state_dict(), f"{path}IA3D_data3velsmall_v.pt")
    torch.save(net2_w.state_dict(), f"{path}IA3D_data3velsmall_w.pt")
    print("Data saved!")

# Usage example:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "E:/VS Code/PINNs/Patient_B/weight/"

# Define the input variables, hyperparameters, etc.
# For simplicity, random tensors are used here as placeholders. Replace with actual data.
x_in = np.random.rand(100, 1)
y_in = np.random.rand(100, 1)
z_in = np.random.rand(100, 1)
xb = np.random.rand(100, 1)
yb = np.random.rand(100, 1)
zb = np.random.rand(100, 1)
ub = np.random.rand(100, 1)
vb = np.random.rand(100, 1)
wb = np.random.rand(100, 1)
xd = np.random.rand(100, 1)
yd = np.random.rand(100, 1)
zd = np.random.rand(100, 1)
ud = np.random.rand(100, 1)
vd = np.random.rand(100, 1)
wd = np.random.rand(100, 1)
xb_inlet = np.random.rand(100, 1)
yb_inlet = np.random.rand(100, 1)
zb_inlet = np.random.rand(100, 1)
ub_inlet = np.random.rand(100, 1)
vb_inlet = np.random.rand(100, 1)
wb_inlet = np.random.rand(100, 1)

# Call the training function
geo_train(device, x_in, y_in, z_in, xb, yb, zb, ub, vb, wb, xd, yd, zd, ud, vd, wd,
          batchsize=512, learning_rate=1e-5, epochs=900, path=path, Flag_batch=True,
          Diff=3.5e-6, rho=1060, Flag_BC_exact=False, Lambda_BC=20., nPt=130, T=0.5,
          xb_inlet=xb_inlet, yb_inlet=yb_inlet, zb_inlet=zb_inlet, ub_inlet=ub_inlet,
          vb_inlet=vb_inlet, wb_inlet=wb_inlet)
