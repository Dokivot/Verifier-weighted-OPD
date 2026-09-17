from __future__ import annotations

import unittest

from opd.runtime_profile import validate_rtx_pro_6000_runtime


class RuntimeProfileTest(unittest.TestCase):
    def test_accepts_rtx_pro_6000_blackwell_profile(self) -> None:
        validate_rtx_pro_6000_runtime(
            torch_version="2.7.1+cu128",
            torch_cuda="12.8",
            driver_version="570.26.00",
            device_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            compute_capability=(12, 0),
            memory_gib=95.0,
            compiled_architectures=["sm_80", "sm_90", "sm_120"],
        )

    def test_rejects_cuda_126_wheel(self) -> None:
        with self.assertRaisesRegex(ValueError, "CUDA 12.8"):
            validate_rtx_pro_6000_runtime(
                torch_version="2.7.1+cu128",
                torch_cuda="12.6",
                driver_version="580.105.08",
                device_name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
                compute_capability=(12, 0),
                memory_gib=95.0,
                compiled_architectures=["sm_120"],
            )

    def test_rejects_ampere_gpu(self) -> None:
        with self.assertRaisesRegex(ValueError, "RTX PRO 6000"):
            validate_rtx_pro_6000_runtime(
                torch_version="2.7.1+cu128",
                torch_cuda="12.8",
                driver_version="580.105.08",
                device_name="NVIDIA A800 80GB PCIe",
                compute_capability=(8, 0),
                memory_gib=79.0,
                compiled_architectures=["sm_80", "sm_120"],
            )


if __name__ == "__main__":
    unittest.main()
