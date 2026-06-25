# Setup Instructions

This page walks you through getting **Unicorn-Hi-C** (the Multicorn / ScUnicorn / 3DUnicorn pipeline) running on your machine. The recommended path uses the prebuilt Docker image so you do not have to install Python, CUDA, or any dependencies by hand. A manual `pip` install is also documented for users who prefer a local environment.

> **Prerequisites**
> - [Git](https://git-scm.com/downloads)
> - [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows/macOS, or Docker Engine on Linux)
> - For GPU acceleration: an NVIDIA GPU with up-to-date drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

---

## 1. Clone the repository

Clone the repository locally and move into the project directory:

```bash
git clone <ANONYMIZED_REPO_URL> && cd Unicorn-Hi-C
```

The repository is organized into three components:

```
Unicorn-Hi-C/
├── Multicorn/     # Multimodal (ATAC + ChIP + RNA) enhancement, primary framework
├── ScUnicorn/     # Unimodal blind super-resolution backbone Multicorn extends
└── 3DUnicorn/     # 3D genome structure reconstruction from enhanced maps
```

---

## 2. Docker setup (recommended)

### 2.1 Pull the image

Pull the Unicorn Docker image from Docker Hub:

```bash
docker pull unicorn:latest
```

This may take a few minutes depending on your connection.

### 2.2 Verify the image

Once the pull finishes, confirm the image is available:

```bash
docker image ls
```

You should see `unicorn` listed in the output.

### 2.3 Run the container

Start the container and mount your current working directory into it so the container can read your data and write results back to the host.

**Linux / macOS (bash):**

```bash
docker run --rm --gpus all -it --name unicorn -v ${PWD}:${PWD} -w ${PWD} unicorn
```

**Windows (PowerShell):**

```powershell
docker run --rm --gpus all -it --name unicorn -v ${PWD}:${PWD} -w ${PWD} unicorn
```

**Windows (Command Prompt):**

```bat
docker run --rm --gpus all -it --name unicorn -v %cd%:%cd% -w %cd% unicorn
```

What the flags do:

| Flag | Purpose |
| --- | --- |
| `--rm` | Removes the container automatically when you exit |
| `--gpus all` | Exposes all host GPUs to the container (omit this if you have no GPU) |
| `-it` | Opens an interactive terminal inside the container |
| `--name unicorn` | Names the container `unicorn` |
| `-v ${PWD}:${PWD}` | Mounts your current directory into the container |
| `-w ${PWD}` | Sets the working directory inside the container |

> **No GPU?** Drop `--gpus all` from the command and the pipeline will run on CPU.

You are now inside the container with the repository mounted, and all dependencies preinstalled. Continue to the [Next Steps](#4-next-steps) below.

---

## 3. Manual setup (without Docker)

If you prefer a local Python environment instead of Docker, create a virtual environment and install the dependencies for the component you want to run. Python 3.9+ is recommended.

```bash
# from the repository root
python -m venv venv

# activate it
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows PowerShell

# install dependencies (example: Multicorn)
cd Multicorn
pip install -r requirements.txt
```

Each component (`Multicorn/`, `ScUnicorn/`, `3DUnicorn/`) ships its own `requirements.txt`, so repeat the install step in whichever directory you intend to use.

---

## 4. Next steps

With the environment ready, follow the component guides to run the pipeline:

1. **Multicorn** (`Multicorn/`): multimodal scHi-C enhancement (primary framework)
2. **ScUnicorn** (`ScUnicorn/`): unimodal blind super-resolution backbone
3. **3DUnicorn** (`3DUnicorn/`): 3D genome structure reconstruction from enhanced maps

The end-to-end flow is: raw scHi-C → **Multicorn** enhancement → **3DUnicorn** 3D reconstruction.

---

## Troubleshooting

- **`docker: command not found`**: Docker is not installed or not on your `PATH`. Install Docker and restart your terminal.
- **`could not select device driver "" with capabilities: [[gpu]]`**: The NVIDIA Container Toolkit is missing or `--gpus all` was used without a GPU. Install the toolkit, or remove `--gpus all` to run on CPU.
- **Permission denied on the mounted volume (Linux)**: Run Docker with the appropriate user, or adjust ownership of the mounted directory on the host.
- **Slow image pull or timeouts**: Retry `docker pull`; partial layers are cached and resumed.
