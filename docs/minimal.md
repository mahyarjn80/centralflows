# Minimal implementation

This page documents the script [minimal.py](/minimal.py), which contains a mimimal implementation of the central flow for gradient descent.  We recommend starting here if you're looking to understand central flows.

The program simultaneously trains a neural network using gradient descent, the GD central flow, and the gradient flow.

By default, the network is a CNN; the dataset is a 4-class, size 1k subset of CIFAR-10; and the loss criterion is MSE.  These settings are hard-coded in for simplicity, but can be easily changed.

To begin, you can try running:
```bash
python minimal.py --lr 0.02 --steps 400 --neigs 3
```
This command specifies that the learning rate should be 0.02, that training should last 400 steps, and that the top 3 Hessian eigenvalues should be tracked.

On a H100, this command takes less than 20 minutes to fully finish.

The script will output the relative path of the output directory where results are stored:
```
output directory: experiments/[random experiment id]
```
In particular, the experiment data will go in the file `experiments/[random experiment id]/data.pt`.  This file will be updated periodically, as the program runs -- you don't need to wait for the experiment to finish to start looking at the experiment data.

You can load the data like so:
```python
from src.utils import load_pytree

data = load_pytree("experiments/aHhdSbfi/data.pt")
```
To plot the loss curves for gradient descent and the central flow, you can do:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(data['discrete']['train_loss'], color="C0", label="gradient descent");
ax.plot(data['central']['predicted_loss'], color='black', label="central flow prediction");
ax.set(xlabel="step", ylabel="train loss", title="train loss curves")
ax.legend();
```
![Train loss curves](/figures/minimal-trainloss.png)

To plot the top Hessian eigenvalues for gradient descent and the central flow, you can do:

### What's _not_ here

The main code `main.py` contains the following additional features that are not found in this minimal implementation:

- **More optimizers**: In addition to gradient descent, the main code implements central flows for Scalar RMSProp and RMSProp.
- **Fancy eigenvalue logic**: In the minimal code, the central flow tracks a pre-specified number of the top Hessian eigenvalues.  If this number is chosen too small, the central flow can 
- Midpoints process 
- more logging
- Efficient data saving
- Checkpointing
