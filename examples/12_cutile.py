# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 12: cuTile
==================

This example shows how to use Nsight Python with cuTile kernels and compare
kernel times with PyTorch for different problem sizes.

New concepts:
- Profiling cuTile kernels for different problem sizes
"""

import math

import cuda.tile as ct
import torch

import nsight


@ct.kernel  # type: ignore[untyped-decorator]
def vec_add_kernel_1d(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, TILE: ct.Constant[int]
) -> None:
    """
    cuTile kernel for 1D element-wise vector addition using direct tiled loads/stores.

    Each block processes a `TILE`-sized chunk of the vectors.
    This approach is efficient when the total dimension is a multiple of `TILE`,
    or when out-of-bounds accesses are implicitly handled by the calling context
    (e.g., by padding or ensuring input sizes match grid dimensions).

    Args:
        a: Input tensor A.
        b: Input tensor B.
        c: Output tensor for the sum (A + B).
        TILE (ct.Constant[int]): The size of the tile (chunk of data) processed by each
                         block. This must be a compile-time constant.
    """
    # Get the global ID of the current block along the first dimension.
    # In a 1D grid, this directly corresponds to the index of the tile.
    bid = ct.bid(0)

    # Load TILE-sized chunks from input vectors 'a' and 'b'.
    # `ct.load` automatically distributes the load operation across the threads
    # within the block, bringing the specified tile of data into shared memory
    # or registers. The `index=(bid,)` specifies which tile to load based on the block ID.
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))

    # Perform the element-wise addition on the loaded tiles.
    # This operation happens in parallel across the threads within the block.
    sum_tile = a_tile + b_tile

    # Store the resulting TILE-sized chunk back to the output vector 'c'.
    # `ct.store` writes the computed tile back to global memory, again
    # distributing the store operation across threads.
    ct.store(c, index=(bid,), tile=sum_tile)


def vec_add_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Helper function to launch the kernel."""
    output = torch.empty_like(x)

    N = x.shape[0]  # Get the total size of the 1D vector

    # Heuristic for TILE size:
    # Choose a power of 2, up to 1024, that is greater than or equal to N.
    # This helps in efficient memory access patterns on the GPU.
    # Handle N=0 gracefully to avoid log2(0) errors.
    TILE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1

    # Calculate the grid dimensions for launching the kernel.
    # `math.ceil(N / TILE)` determines the number of blocks needed to cover
    # the entire vector. Each block processes a `TILE`-sized chunk.
    grid = (math.ceil(N / TILE), 1, 1)  # (blocks_x, blocks_y, blocks_z)

    ct.launch(
        torch.cuda.current_stream(), grid, vec_add_kernel_1d, (x, y, output, TILE)
    )

    return output


# Define sizes to test
sizes = [2**i for i in range(21, 26)]


@nsight.analyze.plot(
    filename="12_cutile.png",
    title="Vector Addition: cuTile/PyTorch",
    ylabel="kernel duration (us)",
)
@nsight.analyze.kernel(
    configs=sizes,
    runs=10,
)
def benchmark_cutile(n: int) -> None:
    """
    Compare cuTile and PyTorch kernel times.

    The plot will show:
    - Y-axis: kernel time
    - X-axis: Problem size (n)
    """
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    # PyTorch
    with nsight.annotate("torch"):
        _ = x + y

    # cuTile kernel
    with nsight.annotate("cuTile"):
        _ = vec_add_1d(x, y)


def main() -> None:
    benchmark_cutile()
    print("✓ cuTile benchmark complete! Check '12_cutile.png'")
    print("\nWhat this example demonstrates:")
    print("\nPlotting cuTile and PyTorch kernel times for different problem sizes!")


if __name__ == "__main__":
    main()
