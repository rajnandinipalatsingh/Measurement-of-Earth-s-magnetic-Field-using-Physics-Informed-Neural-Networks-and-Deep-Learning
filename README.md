# Physics-Informed Neural Network for Earth's Horizontal Magnetic Field

## Overview

This project uses a Physics-Informed Neural Network (PINN) to analyze experimental data from a tangent galvanometer and estimate the horizontal component of Earth's magnetic field, \(B_H\).

The experimental PDF contains measurements for 50 turns and 500 turns of a coil with a radius of 6.8 cm. The measurements include current, compass deflection, reversed-current deflection, and average deflection angle. 

## Project Objective

The main objectives of this project are:

1. Automatically extract experimental data from the PDF.
2. Convert the extracted data into PyTorch tensors.
3. Train a Physics-Informed Neural Network using the experimental data.
4. Apply a physics-based constraint during training.
5. Optimize the slope parameter \(m\).
6. Calculate the horizontal component of Earth's magnetic field \(B_H\).
7. Display the final magnetic field in Tesla, milliTesla, and microTesla.

## Experimental Data

The supplied PDF contains two experimental datasets.

### 50 Turns of Coil

| Current (mA) | Average θ (°) |
| -----------: | ------------: |
|           30 |          19.5 |
|           40 |          24.5 |
|           54 |          31.5 |
|           62 |          35.5 |
|           76 |          40.5 |
|           85 |            44 |
|          113 |          51.5 |
|          137 |          56.5 |
|          167 |            62 |
|          226 |            68 |

### 500 Turns of Coil

| Current (mA) | Average θ (°) |
| -----------: | ------------: |
|            3 |            18 |
|          4.2 |            24 |
|          5.6 |            30 |
|          6.9 |            36 |
|          8.8 |          42.5 |
|         10.4 |            48 |
|         13.1 |            54 |
|           16 |            59 |
|         19.3 |          63.5 |
|         25.4 |            69 |

These values are taken from the experimental tables in the supplied PDF. 

## Physics Behind the Project

The tangent galvanometer relationship used in the experiment is:

$$
B_H = \frac{B_i}{\tan(\theta)}
$$

The magnetic field produced by the coil is:

$$
B_i = \frac{\mu_0NI}{2R}
$$

Therefore:

$$
B_H = \frac{\mu_0NI}{2R\tan(\theta)}
$$

where:

* \(B_H\) = horizontal component of Earth's magnetic field
* \(B_i\) = magnetic field produced by the coil
* \(\mu_0\) = magnetic permeability of free space
* \(N\) = number of turns
* \(I\) = current through the coil
* \(R\) = radius of the coil
* \(\theta\) = compass deflection angle

The experimental setup uses a coil radius of 6.8 cm. 

Rearranging the equation gives:

$$
I =
\frac{2RB_H}{\mu_0N}\tan(\theta)
$$

This shows that current \(I\) is related to \(\tan(\theta)\), which is the relationship learned by the PINN.

## PINN Architecture

The neural network consists of three fully connected layers:

```text
Input
  |
  v
Linear(1 → 20)
  |
  v
Tanh
  |
  v
Linear(20 → 20)
  |
  v
Tanh
  |
  v
Linear(20 → 1)
  |
  v
Output
```

The model also contains a trainable physical parameter:

```python
self.m = nn.Parameter(torch.tensor(0.00887))
```

The parameter is initialized using the median of the available experimental ratios.

## Loss Function

The model uses two components in its loss function.

### Data Loss

The predicted current is compared with the experimental current using Mean Squared Error:

$$
L_{data} = MSE(I_{pred}, I_{true})
$$

In the code:

```python
loss_data = criterion(I_pred, I_true_flat)
```

### Physics Loss

PyTorch automatic differentiation is used to calculate the derivative of the predicted output with respect to the input.

The physics constraint is:

$$
\frac{dI}{d(\tan\theta)} \approx m
$$

The physics loss is:

```python
loss_physics = criterion(
    gradients,
    model.m * torch.ones_like(gradients)
)
```

The total loss is:

$$
L = L_{data} + 4L_{physics}
$$

implemented as:

```python
loss = loss_data + 4.0 * loss_physics
```

## Automatic PDF Data Extraction

Instead of manually entering the experimental measurements, the project uses `pdfplumber` to extract the tables from the PDF.

```python
import pdfplumber

pdf_file = "experimentaldata.pdf"

with pdfplumber.open(pdf_file) as pdf:
    tables = pdf.pages[0].extract_tables()
```

The extracted values are then converted into PyTorch tensors and passed to the PINN.

This allows the experimental PDF to act as the input data source for the project.

## Technologies Used

* Python
* PyTorch
* NumPy
* pdfplumber

## Installation

Install the required packages using:

```bash
pip install torch numpy pdfplumber
```

## Project Structure

```text
project/
│
├── experimentaldata.pdf
├── main.py
└── README.md
```

`experimentaldata.pdf` contains the experimental measurements.

`main.py` contains the PDF extraction, PINN model, training process, and magnetic-field calculation.

`README.md` contains the project documentation.

## Running the Project

Place the experimental PDF in the same directory as the Python program:

```text
experimentaldata.pdf
```

Then execute:

```bash
python main.py
```

The program automatically extracts the experimental data, initializes the PINN, trains the model, and calculates the final value of \(B_H\).

## Training

The model uses the Adam optimizer:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
```

A StepLR learning-rate scheduler is also used:

```python
scheduler = StepLR(
    optimizer,
    step_size=1000,
    gamma=0.5
)
```

The model is trained for 4215 epochs.

Training information is displayed every 1000 epochs, including:

```text
Epoch
Loss
Slope (m)
```

## Calculation of Earth's Magnetic Field

After training, the optimized slope is used in:

$$
B_H = \frac{\mu_0Nm}{2R}
$$

The program uses:

```python
mu_0 = 4 * np.pi * 1e-7
N = 500
R = 0.068
```

The final result is displayed as:

```text
Tesla
milliTesla (mT)
microTesla (μT)
```

## Experimental Results

The PDF's calculated values show approximately 39–42 μT for the 50-turn measurements and approximately 42.6–45.0 μT for the 500-turn measurements. 

These experimental values provide a reference for evaluating the PINN's optimized result.

## Key Features

* Automatic extraction of experimental data from PDF
* Physics-Informed Neural Network
* PyTorch implementation
* Automatic differentiation
* Experimental data loss
* Physics-based loss
* Trainable physical parameter
* Adam optimization
* Learning-rate scheduling
* Automatic calculation of \(B_H\)
* Output in Tesla, mT, and μT

## Important Note

The experimental PDF contains separate datasets for 50-turn and 500-turn coils. Since the number of turns \(N\) is part of the tangent-galvanometer equation, these datasets represent different experimental configurations and should be interpreted accordingly. 

The current implementation is designed to preserve the original PINN code structure while replacing manually entered experimental data with automatic PDF extraction.
