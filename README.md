Parallel Conway’s Game of Life (MPI + 2-D Decomposition)

This project implements a scalable, parallel version of Conway’s Game of Life using MPI and a 2-D Cartesian domain decomposition.
It supports extremely large grids (up to 16,384 × 16,384), multiple Gosper glider guns, halo exchange, visualization, and correctness verification via a global checksum.

The implementation uses mpi4py, NumPy, and a custom MPI halo-exchange scheme that supports toroidal boundaries.
Features

2-D MPI Cartesian communicator (MPI.Cart_create)

Even domain decomposition with per-rank 1-cell halos

Strictly verified neighbor exchange (rows, columns, corners)

Fully periodic (toroidal) grid

Multiple Gosper glider guns at user-defined coordinates

Optional visualization + GIF export

Global correctness validation using a Blake2b checksum

Strong-scaling performance:
~5.6 billion cell updates/sec on 32 ranks
