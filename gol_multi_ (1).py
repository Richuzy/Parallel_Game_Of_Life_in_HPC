#!/usr/bin/env python3
# 2-D MPI Game of Life (toroidal)
# Multi-Gosper-gun toy: empty background + N guns you specify
# - 2-D Cartesian topology, halo rows/cols + 4 corners (Send/Recv)
# - Init: empty + multiple Gosper glider guns (no random cells)
# - Small defaults: nx=128, ny=128, steps=240
# - Rank 0 can dump PNG frames and stitch a GIF
#
# Examples:
#   # two guns, fixed coords
#   mpirun -np 4 python -m mpi4py gol_toy_multi_gun.py \
#       --nx 128 --ny 128 --steps 240 \
#       --guns "20,10; 70,40" --viz_every 5 --gif
#
#   # grid placement of guns (2x2 tiling)
#   mpirun -np 4 python -m mpi4py gol_toy_multi_gun.py \
#       --nx 192 --ny 256 --steps 300 \
#       --gun_grid 2 2 --gun_dx 80 --gun_dy 100 --viz_every 5 --gif
#
# After run, rank 0 prints: Alive, Time, Steps/s, MCellUpdates/s, [CHECKSUM]
# Frames: --outdir (default frames_multi). GIF: --gif

from mpi4py import MPI
import numpy as np
import argparse, os, hashlib

# ---------- pattern placement ----------
def place_gosper_gun(full, x0, y0):
    """Place a Gosper glider gun with its canonical rightward orientation."""
    gun = [
        (5,1),(5,2),(6,1),(6,2),
        (5,11),(6,11),(7,11),(4,12),(8,12),(3,13),(9,13),(3,14),(9,14),
        (6,15),(4,16),(8,16),(5,17),(6,17),(7,17),(6,18),
        (3,21),(4,21),(5,21),(3,22),(4,22),(5,22),(2,23),(6,23),
        (1,25),(2,25),(6,25),(7,25),
        (3,35),(4,35),(3,36),(4,36)
    ]
    nx, ny = full.shape
    for dx, dy in gun:
        full[(x0+dx) % nx, (y0+dy) % ny] = 1

# ---------- util ----------
def split_even(n, parts, i):
    base = n // parts
    extra = n % parts
    ln = base + (1 if i < extra else 0)
    start = i * base + min(i, extra)
    return ln, start

def parse_guns_arg(guns_str):
    """Parse --guns 'x1,y1; x2,y2; ...' -> list of (x,y) ints."""
    pts = []
    if not guns_str:
        return pts
    for token in guns_str.split(';'):
        token = token.strip()
        if not token:
            continue
        x_s, y_s = token.split(',')
        pts.append((int(x_s.strip()), int(y_s.strip())))
    return pts

def parse_args():
    p = argparse.ArgumentParser(description="2-D MPI Life: multiple Gosper guns, empty background")
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=128)
    p.add_argument("--steps", type=int, default=240)
    # Either list explicit gun coordinates OR generate a grid of guns:
    p.add_argument("--guns", type=str, default="", help="semicolon-separated x,y list, e.g. '20,10; 80,40'")
    p.add_argument("--gun_grid", nargs=2, type=int, default=None, help="tile guns in a grid: GX GY")
    p.add_argument("--gun_dx", type=int, default=64, help="x spacing between guns for --gun_grid")
    p.add_argument("--gun_dy", type=int, default=64, help="y spacing between guns for --gun_grid")
    p.add_argument("--gun_x0", type=int, default=16, help="grid origin x for --gun_grid")
    p.add_argument("--gun_y0", type=int, default=16, help="grid origin y for --gun_grid")
    p.add_argument("--viz_every", type=int, default=0, help="save a frame every N steps (0=off)")
    p.add_argument("--outdir", type=str, default="frames_multi")
    p.add_argument("--gif", action="store_true", help="write a GIF at end (requires imageio, matplotlib)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

def maybe_make_frame(full, outdir, step, dims):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    plt.figure(figsize=(5,5), dpi=120)
    plt.imshow(full, cmap="gray", interpolation="nearest", origin="upper")
    # draw rank gridlines (red)
    px, py = dims
    nx, ny = full.shape
    y0 = 0
    for j in range(py):
        ln_y, _ = split_even(ny, py, j); y0 += ln_y
        if j < py-1:
            plt.axvline(y0-0.5, color="red", linewidth=0.8)
    x0 = 0
    for i in range(px):
        ln_x, _ = split_even(nx, px, i); x0 += ln_x
        if i < px-1:
            plt.axhline(x0-0.5, color="red", linewidth=0.8)
    plt.title(f"Step {step}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"step_{step:05d}.png"))
    plt.close()

# ---------- main ----------
def main():
    args = parse_args()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # 2-D cartesian communicator with torus wrap
    dims = MPI.Compute_dims(size, 2)
    cart = comm.Create_cart(dims=dims, periods=(True, True), reorder=False)
    rank = cart.rank
    px, py = dims
    ix, iy = cart.Get_coords(rank)

    nx, ny, steps = args.nx, args.ny, args.steps
    ln_x, gx0 = split_even(nx, px, ix)
    ln_y, gy0 = split_even(ny, py, iy)

    curr = np.zeros((ln_x + 2, ln_y + 2), dtype=np.uint8)
    nxt  = np.zeros_like(curr)

    if args.debug and size <= 16:
        cart.Barrier()
        for r in range(size):
            if r == rank:
                print(f"[DBG] rank={rank} coords=({ix},{iy}) tile=({ln_x}x{ln_y}) "
                      f"offset=({gx0},{gy0}) dims={px}x{py}", flush=True)
            cart.Barrier()

    # Build initial world on rank 0: empty + multiple guns
    # Users can specify explicit list (--guns) or a grid (--gun_grid)
    guns = parse_guns_arg(args.guns)
    if args.gun_grid is not None:
        GX, GY = args.gun_grid
        for gx in range(GX):
            for gy in range(GY):
                x = (args.gun_x0 + gx * args.gun_dx) % nx
                y = (args.gun_y0 + gy * args.gun_dy) % ny
                guns.append((x, y))

    if rank == 0:
        full0 = np.zeros((nx, ny), dtype=np.uint8)
        # place all requested guns
        for (x0, y0) in guns:
            place_gosper_gun(full0, x0, y0)
        # scatter tiles
        for r in range(px * py):
            rx, ry = cart.Get_coords(r)
            tx, gx = split_even(nx, px, rx)
            ty, gy = split_even(ny, py, ry)
            tile = full0[gx:gx+tx, gy:gy+ty]
            if r == 0:
                curr[1:-1, 1:-1] = tile
            else:
                cart.Send(tile.ravel(), dest=r, tag=100)
    else:
        buf = np.empty(ln_x * ln_y, dtype=np.uint8)
        cart.Recv(buf, source=0, tag=100)
        curr[1:-1, 1:-1] = buf.reshape(ln_x, ln_y)

    # neighbor ranks
    up,   down  = cart.Shift(0, 1)
    left, right = cart.Shift(1, 1)

    cart.Barrier()
    t0 = MPI.Wtime()

    for s in range(steps+1):
        # Optional visualization: gather full to rank 0 and write frame
        if args.viz_every > 0 and (s % args.viz_every == 0 or s == steps):
            send_buf = curr[1:-1, 1:-1].ravel()
            if rank == 0:
                full = np.empty((nx, ny), dtype=np.uint8)
                full[gx0:gx0+ln_x, gy0:gy0+ln_y] = curr[1:-1, 1:-1]
                for r in range(1, px*py):
                    rx, ry = cart.Get_coords(r)
                    tx, gx = split_even(nx, px, rx)
                    ty, gy = split_even(ny, py, ry)
                    rb = np.empty(tx*ty, dtype=np.uint8)
                    cart.Recv(rb, source=r, tag=200)
                    full[gx:gx+tx, gy:gy+ty] = rb.reshape(tx, ty)
                maybe_make_frame(full, args.outdir, s, dims)
            else:
                cart.Send(send_buf, dest=0, tag=200)

        if s == steps:
            break

        # --- Halo exchange ---
        # rows
        cart.Sendrecv(curr[1, 1:-1],   dest=up,   sendtag=10,
                      recvbuf=curr[-1, 1:-1], source=down, recvtag=10)
        cart.Sendrecv(curr[-2, 1:-1],  dest=down, sendtag=11,
                      recvbuf=curr[0,  1:-1], source=up,   recvtag=11)
        # columns (pack contiguous)
        left_col   = np.ascontiguousarray(curr[1:-1, 1])
        right_col  = np.ascontiguousarray(curr[1:-1, -2])
        recv_left  = np.empty_like(left_col)
        recv_right = np.empty_like(right_col)
        cart.Sendrecv(left_col,  dest=left,  sendtag=12,
                      recvbuf=recv_right, source=right, recvtag=12)
        cart.Sendrecv(right_col, dest=right, sendtag=13,
                      recvbuf=recv_left,  source=left,  recvtag=13)
        curr[1:-1,  0] = recv_left
        curr[1:-1, -1] = recv_right
        # corners (4 single cells)
        px_, py_ = dims
        ul = cart.Get_cart_rank(((ix - 1) % px_, (iy - 1) % py_))
        ur = cart.Get_cart_rank(((ix - 1) % px_, (iy + 1) % py_))
        dl = cart.Get_cart_rank(((ix + 1) % px_, (iy - 1) % py_))
        dr = cart.Get_cart_rank(((ix + 1) % px_, (iy + 1) % py_))
        tmp = np.empty(1, dtype=np.uint8)
        cart.Sendrecv(np.array([curr[-2, -2]], dtype=np.uint8), dest=dr, sendtag=20,
                      recvbuf=tmp,                                source=ul, recvtag=20)
        curr[0, 0] = tmp[0]
        cart.Sendrecv(np.array([curr[-2,  1]], dtype=np.uint8), dest=dl, sendtag=21,
                      recvbuf=tmp,                                source=ur, recvtag=21)
        curr[0, -1] = tmp[0]
        cart.Sendrecv(np.array([curr[ 1, -2]], dtype=np.uint8), dest=ur, sendtag=22,
                      recvbuf=tmp,                                source=dl, recvtag=22)
        curr[-1, 0] = tmp[0]
        cart.Sendrecv(np.array([curr[ 1,  1]], dtype=np.uint8), dest=ul, sendtag=23,
                      recvbuf=tmp,                                source=dr, recvtag=23)
        curr[-1, -1] = tmp[0]

        # --- Update rule (B3/S23) ---
        A = curr
        N = (A[0:-2, 0:-2] + A[0:-2, 1:-1] + A[0:-2, 2:] +
             A[1:-1, 0:-2] +                 A[1:-1, 2:] +
             A[2:,   0:-2] + A[2:,   1:-1] + A[2:,   2:])
        M = A[1:-1, 1:-1]
        nxt[1:-1, 1:-1] = ((M & ((N == 2) | (N == 3))) | ((M == 0) & (N == 3))).astype(np.uint8)
        curr, nxt = nxt, curr
        nxt[1:-1, 1:-1].fill(0)

    cart.Barrier()
    t1 = MPI.Wtime()
    elapsed = t1 - t0

    # Global alive + checksum over full grid
    alive_local = int(curr[1:-1, 1:-1].sum())
    alive_total = cart.allreduce(alive_local, op=MPI.SUM)

    send_buf = curr[1:-1, 1:-1].ravel()
    if rank == 0:
        full_final = np.empty((nx, ny), dtype=np.uint8)
        full_final[gx0:gx0+ln_x, gy0:gy0+ln_y] = curr[1:-1, 1:-1]
        for r in range(1, px * py):
            rx, ry = cart.Get_coords(r)
            tx, gx = split_even(nx, px, rx)
            ty, gy = split_even(ny, py, ry)
            rb = np.empty(tx*ty, dtype=np.uint8)
            cart.Recv(rb, source=r, tag=999)
            full_final[gx:gx+tx, gy:gy+ty] = rb.reshape(tx, ty)
        chk = hashlib.blake2b(full_final.tobytes(), digest_size=16).hexdigest()
    else:
        cart.Send(send_buf, dest=0, tag=999)
        chk = None

    max_time = cart.allreduce(elapsed, op=MPI.MAX)

    if rank == 0:
        cells = nx * ny
        mcu_s = (cells / 1e6 * steps) / max_time if max_time > 0 else float("inf")
        sps   = steps / max_time if max_time > 0 else float("inf")
        print(f"[OK] ranks={size} px={px} py={py} nx={nx} ny={ny} steps={steps}")
        print(f"Alive={alive_total}  Time={max_time:.6f}s  Steps/s={sps:.2f}  MCellUpdates/s={mcu_s:.2f}")
        print(f"[CHECKSUM] {chk}")

        if args.gif and args.viz_every > 0:
            try:
                import imageio.v2 as imageio, glob
                paths = sorted(glob.glob(os.path.join(args.outdir, "step_*.png")))
                if paths:
                    imgs = [imageio.imread(p) for p in paths]
                    gif_path = os.path.join(args.outdir, "life_multi_guns.gif")
                    imageio.mimsave(gif_path, imgs, fps=max(2, 12 // max(1, args.viz_every)))
                    print(f"[GIF] wrote {gif_path}")
                else:
                    print("[GIF] no frames found (viz disabled or no steps matched)")
            except Exception as e:
                print(f"[GIF] skipped ({e})")

if __name__ == "__main__":
    main()
