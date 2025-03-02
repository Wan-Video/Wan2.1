# Installation Guide

## Install with pip

```bash
pip install .
pip install .[dev]  # Installe aussi les outils de dev
```

## Install with Poetry

Ensure you have [Poetry](https://python-poetry.org/docs/#installation) installed on your system.

To install all dependencies:

```bash
poetry install
```

### Handling `flash-attn` Installation Issues

If `flash-attn` fails due to **PEP 517 build issues**, you can try one of the following fixes.

#### No-Build-Isolation Installation (Recommended)
```bash
poetry run pip install --upgrade pip setuptools wheel
poetry run pip install flash-attn --no-build-isolation
poetry install
```

#### Install from Git (Alternative)
```bash
poetry run pip install git+https://github.com/Dao-AILab/flash-attention.git
```

---

### Running the Model

Once the installation is complete, you can run **Wan2.1** using:

```bash
poetry run python generate.py --task t2v-14B --size '1280x720' --ckpt_dir ./Wan2.1-T2V-14B --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

#### Test
```bash
pytest tests/
```
#### Format
```bash
black .
isort .
```
