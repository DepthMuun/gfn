import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Force sequential compilation to prevent NVCC Out-Of-Memory (LLVM ERROR)
os.environ["MAX_JOBS"] = "1"

# Base directory
csrc_dir = os.path.dirname(os.path.abspath(__file__))

sources = [
    os.path.join(csrc_dir, "extension.cpp"),
    os.path.join(csrc_dir, "losses",      "toroidal.cu"),
    os.path.join(csrc_dir, "geometry",    "low_rank.cu"),
    os.path.join(csrc_dir, "integrators", "integrators.cpp"),
]

# ── Compiler flags ────────────────────────────────────────────────────────────
# --use_fast_math  : enables sincosf, __expf, tanhf, __sqrtf intrinsics
# --ptxas-options  : prints register / smem usage per kernel (useful for tuning)
# -lineinfo        : embeds source line info for Nsight profiler
extra_compile_args = {
    "cxx": ["-O3"],
    "nvcc": [
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
        "-allow-unsupported-compiler",
    ],
}

if os.name == "nt":
    # MSVC: /O2 is the closest to -O3; /std:c++17 required for if constexpr
    extra_compile_args["cxx"] = ["/O2", "/std:c++17"]

setup(
    name="gfn_cuda",
    ext_modules=[
        CUDAExtension(
            name="gfn_cuda",
            sources=sources,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
