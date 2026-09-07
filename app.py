import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
import pdfplumber

pdf_file = "experimentaldata.pdf"

with pdfplumber.open(pdf_file) as pdf:
    tables = pdf.pages[0].extract_tables()
    table_50 = tables[0]
    data_50 = []
    for row in table_50:
        try:
            current = float(row[1])
            left = float(row[2])
            right = float(row[3])
            reverse_left = float(row[4])
            reverse_right = float(row[5])

            data_50.append([
                current,
                left,
                right,
                reverse_left,
                reverse_right
            ])
        except (ValueError, TypeError, IndexError):
            continue

    table_500 = tables[1]
    data_500 = []

    for row in table_500:
        try:
            current = float(row[1])
            left = float(row[2])
            right = float(row[3])
            reverse_left = float(row[4])
            reverse_right = float(row[5])

            data_500.append([
                current,
                left,
                right,
                reverse_left,
                reverse_right
            ])

        except (ValueError, TypeError, IndexError):
            continue

I_true = torch.tensor(data_50, dtype=torch.float32)
tan_theta = torch.tensor(data_500, dtype=torch.float32)

print("\n=== Data extracted from PDF ===")
print("\n50 Turns:")
print(I_true)
print("\n500 Turns:")
print(tan_theta)

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()

        # Neural network layers
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)

        self.m = nn.Parameter(torch.tensor(0.00887))

        # Physics parameter initialization
        with torch.no_grad():

            ratios = I_true.flatten() / tan_theta.flatten()
            valid_ratios = ratios[torch.isfinite(ratios)]
            initial_m = torch.median(valid_ratios)
            self.m = nn.Parameter(
                initial_m.clone().detach()
            )


    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

model = PINN()

print("\nModel parameters:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")


criterion = nn.MSELoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)

print("\nOptimiser initialised successfully")

for epoch in range(4215):
    optimizer.zero_grad()
    tan_theta_flat = tan_theta.view(-1, 1)
    I_true_flat = I_true.view(-1, 1)
    I_pred = model(tan_theta_flat)
    loss_data = criterion(
        I_pred,
        I_true_flat
    )

    tan_theta_flat.requires_grad_(True)
    I_pred_physics = model(tan_theta_flat)

    gradients = torch.autograd.grad(
        outputs=I_pred_physics,
        inputs=tan_theta_flat,
        grad_outputs=torch.ones_like(I_pred_physics),
        create_graph=True
    )[0]

    loss_physics = criterion(
        gradients,
        model.m * torch.ones_like(gradients)
    )

    loss = loss_data + 4.0 * loss_physics

    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 1000 == 0:
        print(
            f"Epoch {epoch:4d}, "
            f"Loss: {loss.item():.2f}, "
            f"Slope (m): {model.m.item():.4f}"
        )


mu_0 = 4 * np.pi * 1e-7
N = 500
R = 0.068
final_m = model.m.item()

B_H_tesla = (
    mu_0 * N * final_m
) / (
    2 * R
)


B_H_microtesla = (
    B_H_tesla * 1e6
)

print("\n=== Final Results ===")
print(f"Optimized slope (m): " f"{final_m:.6f}")
print(f"Magnetic field strength (B_H):")
print(f"{B_H_tesla:.8f} Tesla")
print(f"{B_H_tesla * 1000:.4f} milliTesla (mT)")
print(f"{B_H_microtesla:.2f} microTesla (μT)")
