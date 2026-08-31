# Environment audit — 2026-08-25

Scope: local setup only. No model API was called, no model weights were
downloaded, and the system Python installation was not modified.

## Host and tools

- OS: Microsoft Windows 11 Education, build 10.0.26200
- Git: 2.53.0.windows.3
- `uv`: 0.12.5, user-scoped executable at
  `C:\Users\cyb\AppData\Local\Microsoft\WinGet\Links\uv.exe`
- System Python (left untouched): 3.13.14
- Project Python managed by `uv`: CPython 3.12.14 in root `.venv`
- Root environment verification: `uv sync --frozen` succeeds
- Starter commit: `16d129859e1f0e281363fb4f5910bcaeea316b10`

## Linux / GPU

- `wsl --status` exits 50 because the Windows Subsystem for Linux optional
  component is not enabled. Enabling it requires an elevated install and a
  reboot, so the reproducible setup continued in native Windows instead.
- GPU: NVIDIA GeForce RTX 4070 Ti, 12,282 MiB
- NVIDIA driver: 591.86
- `nvidia-smi`: available
- `nvcc`: not available; no system CUDA toolkit was installed

The missing WSL component and `nvcc` do not block shipped-data analysis,
plotting, mock experiments, or provider-hosted behavioral sampling. They must
be revisited before a local GPU interpretation rehearsal, without modifying the
behavioral environment.

## Storage snapshot

- C: 55.38 GB free
- D: 54.87 GB free

These are time-local observations, not guaranteed capacity for future model
caches.

## Reproduction commands

After opening a fresh terminal so the user-scoped `uv` PATH update is visible:

```powershell
uv sync --frozen
uv run python --version
uv run python -m unittest discover -s tests -v
```
