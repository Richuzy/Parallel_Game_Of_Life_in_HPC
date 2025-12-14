Parallel Conway’s Game of Life (MPI + 2-D Decomposition)

This repository contains a parallel implementation of Conway’s Game of Life using MPI, NumPy, and a 2-D Cartesian domain decomposition.

The program simulates Life on very large grids (up to 16,384 × 16,384), supports multiple Gosper glider guns, performs halo exchange between MPI ranks, and computes a global checksum to verify correctness.

1. Model Overview

Global grid size: up to 16,384 × 16,384

Parallelization: MPI 2-D Cartesian communicator

Boundary type: toroidal (wrap-around)

Local storage per rank: sub-grid + 1-cell halo

Visualization: optional PNG frames or GIF

Correctness check: Blake2b checksum

Game Rules (B3/S23)

Each cell updates based on its 8 neighbors:

A live cell survives if it has 2 or 3 neighbors

A dead cell becomes alive if it has exactly 3 neighbors

The update rule can be written as:

Next
(
𝑀
,
𝑁
)
=
{
1
	
if 
(
𝑀
=
1
∧
𝑁
∈
{
2
,
3
}
)
∨
(
𝑀
=
0
∧
𝑁
=
3
)
,


0
	
otherwise
.
Next(M,N)={
1
0
	​

if (M=1∧N∈{2,3})∨(M=0∧N=3),
otherwise.
	​

Parallel Domain Decomposition

The total grid is divided into a Px × Py MPI process grid.
Each rank stores:

a tile of size 
(
𝑛
𝑥
/
𝑃
𝑥
)
×
(
𝑛
𝑦
/
𝑃
𝑦
)
(nx/Px)×(ny/Py)

a 1-cell halo on all sides

neighbors determined using MPI.Cart_shift

Halo exchange includes:

top / bottom rows

left / right columns

four corner cells

This ensures correct neighbor counting at tile boundaries.

Gosper Glider Guns

The simulation supports:

explicitly defined gun coordinates, e.g.
"20,10; 16000,16000"

grid-based gun placement using --gun_grid GX GY

These patterns generate gliders that move across MPI rank boundaries, making them ideal for correctness testing.

2. Requirements

Python 3.x

NumPy

mpi4py

(Optional) Matplotlib

(Optional) ImageIO

You can install dependencies with:

pip install numpy mpi4py matplotlib imageio

3. Running the Simulation
Basic CPU run
mpirun -np 4 python -m mpi4py gol_multi.py \
    --nx 128 --ny 128 --steps 240

With two Gosper guns
mpirun -np 4 python -m mpi4py gol_multi.py \
    --nx 128 --ny 128 --steps 240 \
    --guns "20,10; 70,40"

Grid-based placement of guns
mpirun -np 4 python -m mpi4py gol_multi.py \
    --nx 192 --ny 256 --steps 300 \
    --gun_grid 2 2 --gun_dx 80 --gun_dy 100

Enabling visualization
--viz_every 5 --gif


This produces PNG frames and an optional GIF inside frames_multi/.

4. Running on an HPC Cluster (SLURM)

Submit the provided SLURM script:

sbatch sbatch.sh


The script runs a large production simulation:

mpirun -np $SLURM_NTASKS python -m mpi4py gol_toy_multi_gun.py \
    --nx 16384 --ny 16384 --steps 2000 \
    --guns "10,10; 16000,16000" \
    --viz_every 0

5. Correctness Verification

The simulation computes a global checksum from the final grid state:

[CHECKSUM] bf359a08639a07aa4684df45fcc38b9e


If different MPI configurations (e.g., 1, 4, 8, 16, 32 ranks) produce the same checksum, the parallel implementation is correct.

6. Performance Summary

Strong-scaling results for a 16,384² grid (2000 steps):

Ranks	Time (s)	MCUP/s	Speedup	Efficiency
8	377.97	1420	1.00×	100%
16	191.34	2806	1.98×	98.8%
32	96.13	5585	3.33×	98.3%

The model achieves near-linear strong scaling up to 32 ranks.

7. File Structure
gol_multi.py       # Main MPI Life simulation
sbatch.sh          # SLURM submission script
README.md
frames_multi/      # Visualization outputs (optional)

8. Future Extensions

Non-blocking communication (Isend/Irecv)

GPU acceleration (CUDA / CuPy)

Bit-packed cell storage for memory efficiency

Faster I/O and real-time visualization
