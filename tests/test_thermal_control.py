"""
Example: Thermal Throttling Test

This example demonstrates the thermal control by running
sustained heavy GPU workload that will heat up the GPU and trigger
thermovision's cooling mechanism.
"""

import torch

import nsight


@nsight.analyze.kernel(
    runs=10,
    thermal_mode="auto",
    combine_kernel_metrics=lambda x, y: x + y,
    output="verbose",
)
def heavy_gemm_kernel(n: int) -> None:
    """
    This performs EXTREME computation:
    - 12288x12288 matrices (144 million elements each)
    - 20 matrix multiplications per iteration
    - Multiple element-wise operations
    """
    # Create HUGE matrices on GPU
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, n, device="cuda", dtype=torch.float32)
    C = torch.randn(n, n, device="cuda", dtype=torch.float32)

    with nsight.annotate("heavy_gemm"):
        result = A

        for i in range(20):

            # Three-way matrix multiplication
            result = torch.matmul(result, B)
            result = torch.matmul(result, C)

            # Dense element-wise operations
            result = torch.sin(result)
            result = torch.cos(result)
            result = torch.tanh(result)

        # Additional final heavy operation
        final = torch.matmul(result, A)
        final = torch.matmul(final, B)

        # Force synchronization
        torch.cuda.synchronize()


def test() -> None:
    heavy_gemm_kernel(12288)
