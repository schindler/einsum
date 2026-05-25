# Minimal Einstein-summation

## Overview

This project is a small CUDA command-line prompt for experimenting with a
minimal hardcoded set of Einstein-summation operations. It loads or creates
tensors, executes supported patterns on the GPU with cuTENSOR 1.3.3, and prints
execution statistics after each operation. Tensor values are shown only when
requested with `show`.

At startup the program prints the selected CUDA GPU name and device ID. It also
creates a cuTENSOR handle and prints the cuTENSOR version so the configured GPU
math library is checked before the prompt starts.

Supported examples:

```text
T = ij->ji A              # transpose
s = ij-> A                # matrix sum
c = ij->j A               # column sum
r = ij->i A               # row sum
C = ij,jk->ik A B         # matrix multiplication
C = ik,kj->ij A B         # matrix multiplication, alternate labels
y = ij,j->i A x           # matrix-vector multiplication
y = ik,k->i A x           # matrix-vector multiplication, alternate labels
d = i,i-> x y             # dot product
O = i,j->ij x y           # outer product
D = abc,cde->abde A B     # rank-3 contraction
```

Tensor creation and file commands:

```text
A = load data/A.txt
A = matrix 2 3 1 2 3 4 5 6
x = tensor 1 3 1 2 3
A = random 2 3
A = ones 2 3
A = zeros 2 3
show A
show A --output=data/A.bin
```

The project depends only on the CUDA runtime and cuTENSOR 1.3.3. cuTENSOR is
used for GPU permutation, reduction, and contraction.

## Code Organization

```bin/```
Placeholder for generated binaries or build artifacts.

```data/```
Starter tensor files. `data/A.txt` and `data/B.txt` are ready for
`ij,jk->ik`.

```lib/```
Placeholder for manually supplied libraries if needed.

```src/```
CUDA source code. `src/main.cu` contains the REPL, tensor file handling, and
cuTENSOR execution helpers.

```Dockerfile```
Linux CUDA 11.4 build environment with cuTENSOR 1.3.3.2 for checking the
project from macOS with a mounted workspace.

```Makefile```
Builds `./einsum` from `src/main.cu`.

```INSTALL```
Linux and Docker build instructions.
