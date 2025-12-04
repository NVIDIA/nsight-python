# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example 7: Triton Integration with Variants
============================================

This example shows how to use Nsight Python with Triton kernels and explore
different block sizes using the variant visualization features.

New concepts:
- Profiling custom Triton kernels with different configurations
- Using `variant_fields` to create separate lines for different parameter values
- Using `variant_annotations` to create separate lines for different kernels
- Comparing performance across block sizes and implementations
"""

import itertools

import torch
import triton
import triton.language as tl

import nsight


# Define a simple Triton vector addition kernel
@triton.jit  # type: ignore[untyped-decorator]
def add_kernel(
    x_ptr: torch.Tensor,
    y_ptr: torch.Tensor,
    output_ptr: torch.Tensor,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    """Triton kernel for vector addition."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int) -> torch.Tensor:
    """Helper function to launch the Triton kernel with configurable block size."""
    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)

    return output


# Define sizes and block sizes to test
sizes = [2**i for i in range(20, 25)]
block_sizes = [256, 512, 1024, 2048]

# Create all combinations
configs = list(itertools.product(sizes, block_sizes))


@nsight.analyze.plot(
    filename="07_triton_minimal.png",
    title="Vector Addition: Triton Speedup vs Block Size",
    ylabel="Speedup vs PyTorch",
    # variant_fields creates separate lines for different block_size values
    # This lets you see how performance varies with block size
    variant_fields=["block_size"],
    # variant_annotations: only include "triton" because torch doesn't have
    # configurable parameters like block_size. We want separate lines for
    # each triton block size configuration.
    variant_annotations=["triton"],
)
@nsight.analyze.kernel(
    configs=configs,
    runs=10,
    # Normalize against torch to show speedup (values > 1 = triton is faster)
    normalize_against="torch",
)
def benchmark_triton_variants(n: int, block_size: int) -> None:
    """
    Compare Triton with different block sizes against PyTorch baseline.

    The plot will show:
    - Y-axis: Speedup relative to PyTorch (normalized_against="torch")
    - Different lines for each block size (from variant_fields)
    - Only triton kernels shown as separate lines (from variant_annotations)
    - X-axis: Problem size (n)
    """
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")

    # PyTorch baseline (block_size doesn't affect this)
    with nsight.annotate("torch"):
        _ = x + y

    # Custom Triton kernel with configurable block size
    with nsight.annotate("triton"):
        _ = triton_add(x, y, block_size)


def main() -> None:
    benchmark_triton_variants()
    print("✓ Triton benchmark complete! Check '07_triton_minimal.png'")
    print("\nWhat this example demonstrates:")
    print("- normalize_against='torch' shows speedup relative to PyTorch baseline")
    print("- variant_fields=['block_size'] creates separate lines for each block size")
    print("- variant_annotations=['triton'] only shows triton variants (not torch)")
    print("  because torch doesn't have tunable parameters like block_size")
    print("\nThe plot shows how each Triton block size configuration performs")
    print("relative to PyTorch across different problem sizes!")


if __name__ == "__main__":
    main()
