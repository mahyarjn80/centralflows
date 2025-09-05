## Complete documentation for main.py

The program `main.py` simultaneously runs a set of _processes_ (e.g. discrete optimizer, central flow, stable flow) on the same neural net objective, starting from the same initialization.  It logs process-specific metrics such as loss and gradient norm, as well as inter-process metrics such as the distance between the processes.

Currently, the code supports three optimization algorithms: gradient descent, Scalar RMSProp, and RMSProp. Under the hood, all three processes are cast as instances of preconditioned gradient descent $w_{t+1} = w_t - P_t^{-1} \nabla L(w_t) $, where, for example, $P_t = \eta^{-1} I$ for vanilla GD.  The _effective sharpness_ is defined as the maximum eigenvalue of the preconditioned Hessian  $P_t^{-1} H(w_t)$, so EOS is characterized by effective sharpness $\approx 2$. 
For vanilla gradient descent, effective sharpness is sharpness $*$  $\eta$.

Note: we use the library [Tyro](https://brentyi.github.io/tyro/) to parse command-line arguments.  This library automatically sets up command-line argument parsing based on [dataclasses](https://docs.python.org/3/library/dataclasses.html) that are defined in the code.

Below is documentation for the arguments and outputs of `main.py`.

**Arguments**:
 - `opt:[gd|scalar-rmsprop|rmsprop]`: the optimizer to use.  Each optimizer requires additional optimizer-specific arguments, e.g. the hyperparameters, detailed [here](#supported-optimizers).
  - `data:[cifar10|sorting]`: the dataset to train on.  Most datasets take additional dataset-specific arguments, defailed [here](#supported-datasets).
  - `arch:[mlp|cnn|resnet|vit|lstm|transformer|mamba]`: the architecture to train.  Most architectures take additional architecture-specific arguments, detailed [here](#supported-architectures).
  - `--runs` (list of strings, required): the set of processes to run.  The options are `discrete`, `midpoint`, `central`, `stable`.  Each process has additional process-specific arguments, detailed [here](#supported-processes).
  - `--eig.*`: Global settings for eigenvalue computations that are shared between processes.  Note that each process allows for overriding these global settings by passing `process.eig.*=...`.
    - `--eig.frequency` (int; default=-1): frequency, in steps, at which to recompute eigenvalues/eigenvectors of the effective Hessian using a warm-started iterative algorithm (-1 means never).
    - `--eig.initial-neigs` (int; default=1): number of top eigenvalues to initially compute.  (If track-threshold is non-None, more eigenvalues may be added later.)
    - `--eig.track-threshold` (float, default=1.5): If set to None, the number of tracked eigenvalues will not be dynamically changed throughout training. If set to non-None, the number of tracked eigenvalues will grow so as to catch all eigenvalues of the effective Hessian that are above this threshold.
    - `--eig.solver` ("lobpcg" | "scipy"; default="lobpcg"): the eigenvalue solver to use.  If set to `lobpcg`, we use an implementation of [LOBPCG](https://en.wikipedia.org/wiki/LOBPCG) which is a translation of [Google's Jax implementation](https://github.com/jax-ml/jax/blob/main/jax/experimental/sparse/linalg.py#L37-L105) to PyTorch.  This version allows for warm-starting.  If set to `scipy`, we use the default Lanczos iteration from Scipy, which does not allow for warm-starting.
    - `--eig.tol` (float; default=1e-10): the numerical tolerance for the eigenvalue computations (only used by LOBPCG).
    - `--eig.chunk-size` (int; default=-1): the maximum number of Hessian-vector products to try to do on the GPU at once.  If set to -1, there is no cap.
  - `--checkpoint.*`: settings for saving checkpoints
    - `--checkpoint.frequency` (int; optional; defaults to -1=never): save checkpoints at this frequency
    - `--checkpoint.steps` (list of ints; optional; defaults to []): save checkpoints at these steps
  - `--load.*`: settings for loading from checkpoints
   - `--load.path` (string; optional): if provided, initialize from the checkpoint saved at this path 
   - `--load.from-process` ("central"|"discrete"|"midpoint"|"stable", default=None): if set to None (the default), each process will load its weights and optimizer state from the same process saved at the checkpoint (i.e. discrete will load from discrete, etc).  But if set to non-None, each process will instead load from the one specified process.
  - `--warm-start` (int, optional, default=0): if set, warm start training by taking this many steps before the main training loop is run
  - `--steps` (int, default=50): how many steps to train for.
  - `--device` (str, default='cuda'): cuda or cpu
  - `--seed` (int, default=0): the random seed
  - `--expid` (str, optional): if provided, use this as the experiment id, which is used to determine the output directory path.  If not provided, the code will set the experiment id to a random UUID.

**Outputs**: 
 - All outputs will be saved in the directory ``{EXPERIMENT_DIR}/{expid}`` where `EXPERIMENT_DIR` defaults to `experiments` but can be overriden using the `EXPERIMENT_DIR` environment variable, and `expid` defaults to a random UUID but can be overriden using the `expid` command-line argument.
 - Experimental data will be saved in [HDF5 format](https://docs.h5py.org/en/stable/) in a file located at `{EXPERIMENT_DIR}/{expid}/data.hdf5`.  The top-level container has one entry for each process (e.g. `discrete`, `central`, etc,) where the process-specific data is stored.  Further, it has entries for cross-process data such as `distance`, which stores distances between the processes, and `delta`, which (if there is a central flow) stores the deltas between the other processes and the central flow.  Most leaves consist of time series.
 - If checkpointing was enabled, the checkpoints will be saved in the subdirectory `{EXPERIMENT_DIR}/{expid}/checkpoints`.

#### Supported optimizers:
The supported optimizers are:
 - Gradient descent ( `opt:gd` ).  Further arguments:
   - `--opt.lr` (float; required): learning rate hyperparameter
 - Scalar RMSProp ( `opt:scalar-rmsprop` ).  Further arguments:
   - `--opt.lr` (float; required) learning rate hyperparameter
   - `--opt.beta2` (float; required) EMA decay rate $\beta_2$ hyperparameter
   - `--opt.eps` (float; optional; defaults to 0) epsilon hyperparameter
   - `--opt.bias-correction` | `--opt.no-bias-correction` (bool; optional; defaults to false) whether to use bias correction
 - RMSProp ( `opt:rmsprop` ).  Further arguments:
   - `--opt.lr` (float; required) learning rate hyperparameter
   - `--opt.beta2` (float; required) EMA decay rate $\beta_2$ hyperparameter
   - `--opt.eps` (float; optional; defaults to 0) epsilon hyperparameter
   - `--opt.bias-correction` | `--opt.no-bias-correction` (bool; optional; defaults to false) whether to use bias correction
 
#### Supported datasets:
The supported datasets are:
 - CIFAR-10 ( `data:cifar10` ).  Further arguments:
   - `--data.criterion` ("ce" | "mse"; required): the loss criterion to train with, i.e. cross-entropy or mean squared error.
   - `--data.n` (int; required): the number of examples for the train set.  These examples will be randomly selected from the full CIFAR-10 dataset.
   - `--data.classes` (either int or the string "binary"; default=10): the number of classes to train on.  If "binary", will perform binary classification (i.e. only one output dimension) using the first two classes.
 - Toy sorting task ( `data:sorting` ) inspired by [Karpathy's](https://github.com/karpathy/minGPT/blob/master/demo.ipynb).  Further arguments:
   - `--data.criterion` ("ce" | "mse"; required): the loss criterion to train with.
   - `--data.n` (int; required): the number of examples for the train set.
   - `--data.n-test` (int; default=1000): the number of examples for the test set.
   - `--data.vocab-size` (int; defaults to 4): the size of the "alphabet" for the toy sorting task.
   - `--data.length` (int; defaults to 8): the length of the strings to sort.

#### Supported architectures:
The supported architectures are:
 - Multi-layered perceptron ( `arch:mlp` )
 - CNN ( `arch:cnn` )
 - ResNet ( `arch:resnet` ):
 - Vision Transformer ( `arch:vit` )
 - LSTM ( `arch.lstm` ) 
 - Sequence transformer ( `arch.transformer` )
 - Mamba ( `arch.mamba` )

 Please see the source code for the architecture-specific arguments.  For example, [this code](https://github.com/locuslab/central_flows/blob/main/src/architectures/simple.py#L11) shows that the CNN takes arguments `--arch.width` (int) and `--arch.activation` (str).

The MLP, CNN, ResNet, and ViT are intended for vision tasks (e.g. CIFAR-10), while the LSTM, sequence transformer, and Mamba are intended for sequence tasks (e.g. sorting).

#### Supported processes:
  - Discrete optimizer (`discrete`): run the discrete optimization algorithm.  Further arguments:
    - `--discrete.eig.*`:  Process-specific overrides for the eigenvalue computation config.  See the above documentation for `eig.*` for more information on these config options.
  - Central flow (`central`): run the central flow.  Further arguments:
    - `--central.nsubsteps` (int; default=4): To discretize the central flow, we take this many "substeps" per unit of time.  In other words, we use Euler's method with `dt = 1/multiplier`.
    - `--central.sdcp-threshold` (float; default=1.95): All tracked eigenvalues of the effective Hessian that are above this threshold will be considered potentially critical and will be thrown into the SDCP.
    - `--central.frequency-in-substeps`/`--central.no-frequency-in-substeps` (bool; default=False):  If set to False (default), the eigenvalue computation frequency will be interpreted in units of steps.  If set to True, it will be interpreted in units of substeps.  Set this to True if you need a higher frequency of eigenvalue computation.  Note that to set this as True, you pass `--central.frequency-in-substeps` (without putting "true" after that), and to set this as False, you pass `--central.no-frequency-in-substeps`.
    - `--central.eig.*`:  Process-specific overrides for the eigenvalue computation config.  See the above documentation for `eig.*` for more information on these config options.
  - Stable flow (`stable`): run the stable flow.  Further arguments:
    - `--stable.max-Seff` (int; default=100): We stop trying to run the central flow when the effective sharpness exceeds this value.
    - `--stable.min-nsubsteps` (int; default=4): To discretize the stable flow, we dynamically adapt the # of substeps based on the current effective sharpness, but we never allow it to be below this number.
    - `--stable.eig.*`:  Process-specific overrides for the eigenvalue computation config.  See the above documentation for `eig.*` for more information on these config options.
  - Midpoints of discrete optimizer (`midpoint`): run the discrete optimization algorithm and treat the midpoints as the iterates. Further arguments:
    - `--midpoint.eig.*`:  Process-specific overrides for the eigenvalue computation config.  See the above documentation for `eig.*` for more information on these config options.


## Gotchas

1. **LayerNorm bug**: for reasons we do not understand, PyTorch autodiff does not correctly compute third derivatives on architectures which include the default PyTorch `LayerNorm` layer (which is implemented in C++).  Strangely, _second_ derivatives (e.g. Hessian eigenvalues) are computed correctly, but not _third_ derivatives (e.g. the gradients of such eigenvalues).  To get around this, for architectures with LayerNorm, we have replaced the default LayerNorm with a different LayerNorm implementation which is written in straight PyTorch rather than C++.
2. **Eig-subfrequency of central flow**: recall that when discretizing the central flow, we split each optimizer step into `multiplier` substeps.  By default, to save on computation, we recompute the top Hessian eigenvectors each `multiplier` substeps (corresponding to one optimizer step) rather than at every substep.  For most architectures, this seems to work fine, but for some (e.g. Mamba with cross-entropy on CIFAR-10), it is necessary to recompute the top Hessian eigenvectors more frequencly, at each substep.  As described above, this can be achieved by passing `--central.eig-subfrequency 1`.

## Notes

- We originally wrote this codebase so as to be able to support both PyTorch and Jax backends.
While we never got around to implementing Jax support, you'll see remnants of this design decision throughout the code -- example, we often use a generic Array type rather than torch.tensor.