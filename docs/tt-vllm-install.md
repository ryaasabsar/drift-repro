# Python 3.12 environment for Tenstorrent vLLM 0.25.1

From the project root:

```bash
bash scripts/install_tt_vllm.sh
source .venv-tt-vllm/activate-tt.sh
```

The script creates `.venv-tt-vllm` with workspace-managed Python 3.12, builds vLLM
0.25.1 with `VLLM_TARGET_DEVICE=empty`, and installs the new `vllm-tt-plugin` from
the immutable commit behind `compat/vllm-0.25.1`. It uses the official install script
and its documented OpenCV/numpy override. This is a different package from the
older `tt-vllm-plugin` inside `tt-inference-server`.

It also installs TTNN 0.77.0, the matching TT-Metal source release, SFPI 7.69.0, CPU PyTorch
2.11.0, torchvision 0.26.0 and Transformers 5.12.1. The latter versions follow
TT-Metal 0.77.0's development requirements. This is an explicit package baseline
for testing, not a vendor-certified pairing for every P150b model. The newer plugin
lists Python 3.12 in its unit-test CI matrix; those tests use TTNN stubs and do not
validate physical devices.

Existing `.venv`, `.venv-sglang`, `.venv-trt` and ROCm environments are unaffected.
The suite already selects `.venv-tt-vllm/bin/python` for TT vLLM. Source the generated
activation file before running it so the TT-Metal paths and client Python are set.

If `.venv-tt-vllm` was previously created with Python 3.10, replace that environment
explicitly (this removes custom packages in that environment):

```bash
bash scripts/install_tt_vllm.sh --recreate
```

To inspect commands or check installed packages:

```bash
bash scripts/install_tt_vllm.sh --dry-run
bash scripts/install_tt_vllm.sh --check
```

If startup reports that `/opt/tenstorrent/sfpi` rejects `-ftt-nttp`,
`-ftt-constinit`, `-ftt-consteval` or `-ftt-no-dyninit`, repair the compiler in
the existing environment:

```bash
bash scripts/install_tt_vllm.sh --toolchain-only
source .venv-tt-vllm/activate-tt.sh
```

This downloads the release-pinned SFPI archive, verifies its SHA256, and installs
it in `.tools/sfpi-7.69.0`. Both the TTNN wheel and TT-Metal source runtime get a
`runtime/sfpi` link; TT-Metal searches there before `/opt/tenstorrent/sfpi`.
Python packages and the system compiler are left intact. The repair requires the
existing TTNN 0.77.0 environment and matching TT-Metal checkout. It compiles a
small Blackhole object using the four flags above without opening a device.
`--check --toolchain-only` verifies this setup without downloading or linking.

The installer requires Linux x86_64 with glibc >=2.34, bash, git, curl, flock,
tar with xz support, and sha256sum.
It downloads source repositories, their submodules and Python packages, without
`sudo`, containers, model weights or accelerator execution. Source checkouts have
fixed commits and are reused only when unchanged. The final checks verify package
versions/dependencies, CPU PyTorch, TTNN imports, plugin discovery and an SFPI
Blackhole compilation probe. A successful
check writes `.venv-tt-vllm/tt-install-report.json` with resolved versions and source
pins. It does not establish model/kernel or long-context correctness.

**Hardware prerequisite:** TT-Metal 0.77.0 documents Blackhole KMD >=2.8.0 and
firmware 19.8.1. The reported `tt-blackhole-03` versions, KMD 2.6.0-rc1 and firmware
19.4.2.0, are below that baseline. Creating a Python environment does not resolve
that mismatch. Before inference, the host provider must supply the required host
stack or a tested alternative software pairing. This script never updates drivers
or firmware and does not reset devices.

References:

- [Plugin compatibility tag](https://github.com/tenstorrent/vllm-tt-plugin/tree/compat/vllm-0.25.1)
- [Python 3.10/3.12 CI](https://github.com/tenstorrent/vllm-tt-plugin/blob/compat/vllm-0.25.1/.github/workflows/ci.yaml)
- [Pinned upstream installer](https://github.com/tenstorrent/vllm-tt-plugin/blob/6d3bb2854f5f8885acc1b12111d929bffdebc36e/docs/install-vllm-tt.sh)
- [TT-Metal 0.77.0 prerequisites](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/INSTALLING.md)
- [TT-Metal 0.77.0 package versions](https://github.com/tenstorrent/tt-metal/blob/v0.77.0/tt_metal/python_env/requirements-dev.txt)
- [SFPI version and archive checksums](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tt_metal/sfpi-version)
- [Compiler selection and required flags](https://github.com/tenstorrent/tt-metal/blob/9f9cd4fd590f4b606bd0981a4fe0b6403eb38ec9/tt_metal/jit_build/build.cpp)
