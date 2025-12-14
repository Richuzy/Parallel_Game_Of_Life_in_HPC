# Parallel Conway’s Game of Life (MPI Implementation)

This repository contains a parallel implementation of Conway’s Game of Life using **MPI**, **NumPy**, and a **2-D Cartesian domain decomposition**.  
It supports **very large grids (up to 16,384 × 16,384)**, performs **halo exchange** between MPI ranks, supports **multiple Gosper glider guns**, provides **visualization**, and verifies correctness using a **global checksum**.

---

## 1. Model Overview

- Grid size supported: `nx, ny ≤ 16384`
- Parallelization: `2-D Cartesian MPI communicator`
- Boundary type: `toroidal (wrap-around)`
- Halo exchange: `rows`, `columns`, and `corners`
- Visualization: PNG frames + optional GIF
- Correctness: global `Blake2b checksum`
- Tested on MPI ranks: `1–32`

### Game of Life Rule (B3/S23)

A cell updates based on its eight neighbors:

\[
\text{Next}(M,N) =
\begin{cases}
1 & \text{if } (M=1 \land N\in\{2,3\}) \lor (M=0 \land N=3),\\[4pt]
0 & \text{otherwise}.
\end{cases}
\]

### Domain Decomposition (MPI)

- A **Px × Py** Cartesian MPI grid is created using `MPI.Cart_create`
- Each rank holds:
  - A local tile of the global grid
  - A **1-cell halo** on each side for neighbor communication
- Halo exchange sends:
  - Top and bottom rows  
  - Left and right columns  
  - Four corners  

This ensures correct neighbor counts along boundaries.

### Gosper Glider Guns

Two placement methods are supported:

1. **Explicit coordinates**  
   Example: `"20,10; 16000,16000"`

2. **Grid-based placement**  

---

## 2. Requirements

- Python 3.x  
- `numpy`  
- `mpi4py`  
- (Optional) `matplotlib` (for PNG visualization)  
- (Optional) `imageio` (for GIF output)

Install dependencies:

```bash
pip install numpy mpi4py matplotlib imageio
