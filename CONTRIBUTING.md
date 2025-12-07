# Contributing to Wan2.1

Thank you for your interest in contributing to Wan2.1! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Quality](#code-quality)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA 11.8+ (for GPU support)
- Git
- Basic knowledge of PyTorch and diffusion models

### Finding Issues to Work On

- Check the [Issues](https://github.com/Kuaishou/Wan2.1/issues) page for open issues
- Look for issues labeled `good first issue` if you're new to the project
- Issues labeled `help wanted` are specifically looking for contributors
- If you want to work on a new feature, please open an issue first to discuss it

## Development Setup

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/Wan2.1.git
cd Wan2.1
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install in development mode**

```bash
pip install -e .[dev]
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Verify installation**

```bash
pytest tests/ -v
python -c "from wan.modules.model import WanModel; print('Import successful')"
```

## Making Changes

### Branch Naming Convention

Create a descriptive branch name following this pattern:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions or modifications

Example:
```bash
git checkout -b feature/add-video-preprocessing
```

### Commit Message Guidelines

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic changes)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(vae): add support for custom temporal compression ratios

Allows users to specify custom temporal compression ratios for VAE
encoding, enabling more flexible video compression strategies.

Closes #123
```

```
fix(attention): resolve NaN values in flash attention backward pass

The gradient computation was producing NaN values when using
bfloat16 precision. Added gradient clipping to stabilize training.

Fixes #456
```

## Code Quality

### Code Style

We use multiple formatters and linters to ensure consistent code quality:

- **YAPF**: Primary code formatter (configured in `.style.yapf`)
- **Black**: Alternative formatter (line length: 100)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

**Before committing, run:**

```bash
# Format code
yapf --in-place --recursive wan/ tests/
isort wan/ tests/

# Check linting
flake8 wan/ tests/

# Type checking
mypy wan/
```

Or use the Makefile:

```bash
make format
```

### Type Hints

- Add type hints to all new functions and methods
- Use `from typing import` for complex types
- For PyTorch tensors, use `torch.Tensor`
- For optional parameters, use `Optional[Type]`

**Example:**

```python
from typing import Optional, Tuple
import torch

def process_video(
    video: torch.Tensor,
    fps: int = 30,
    output_path: Optional[str] = None
) -> Tuple[torch.Tensor, dict]:
    """Process a video tensor.

    Args:
        video: Input video tensor of shape (T, C, H, W)
        fps: Frames per second
        output_path: Optional path to save processed video

    Returns:
        Processed video tensor and metadata dictionary
    """
    ...
```

### Docstrings

Use Google-style docstrings for all public functions, classes, and methods:

```python
def encode_video(
    self,
    video: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Encode video to latent space using VAE.

    Args:
        video: Input video tensor of shape (B, C, T, H, W)
        normalize: Whether to normalize the input to [-1, 1]

    Returns:
        Latent tensor of shape (B, Z, T', H', W')

    Raises:
        ValueError: If video dimensions are invalid
        RuntimeError: If encoding fails
    """
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run tests matching a pattern
pytest tests/ -k "test_attention" -v

# Run with coverage
pytest tests/ --cov=wan --cov-report=html

# Skip slow tests
pytest tests/ -m "not slow"

# Skip CUDA tests (for CPU-only testing)
pytest tests/ -m "not cuda"
```

### Writing Tests

- Write tests for all new features and bug fixes
- Place unit tests in `tests/test_<module>.py`
- Place integration tests in `tests/test_pipelines.py`
- Use pytest fixtures from `tests/conftest.py`
- Mark slow tests with `@pytest.mark.slow`
- Mark CUDA tests with `@pytest.mark.cuda`

**Example test:**

```python
import pytest
import torch
from wan.modules.model import WanModel

class TestWanModel:
    def test_model_forward_pass(self, sample_config_1_3b, device, dtype):
        """Test that model forward pass produces correct output shape."""
        model = WanModel(**sample_config_1_3b).to(device).to(dtype)
        model.eval()

        # Create dummy inputs
        batch_size = 2
        x = torch.randn(batch_size, 4, 16, 16, 16, device=device, dtype=dtype)
        # ... other inputs

        with torch.no_grad():
            output = model(x, t, y, mask, txt_fea)

        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
```

### Test Coverage

- Aim for >80% code coverage for new code
- Critical paths (model forward pass, VAE encode/decode) should have >95% coverage
- Run coverage reports: `pytest tests/ --cov=wan --cov-report=term-missing`

## Documentation

### Code Documentation

- Add docstrings to all public APIs
- Update README.md if you add new features
- Add inline comments for complex algorithms
- Update type hints

### User Documentation

- Update README.md examples if you change public APIs
- Add usage examples for new features
- Update INSTALL.md if you change dependencies

## Pull Request Process

1. **Before submitting:**
   - Ensure all tests pass locally
   - Run code formatters and linters
   - Update documentation
   - Add/update tests for your changes
   - Rebase your branch on latest main

2. **Submit your PR:**
   - Write a clear title following conventional commits
   - Fill out the PR template completely
   - Reference related issues (e.g., "Closes #123")
   - Add screenshots/videos for UI changes
   - Request review from maintainers

3. **PR template:**

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] Added new tests
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
```

4. **After submission:**
   - Respond to review comments promptly
   - Make requested changes
   - Keep PR updated with main branch
   - Squash commits if requested

## Release Process

Releases are managed by project maintainers. The process includes:

1. Version bump in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Build and upload to PyPI (if applicable)
5. Create GitHub release with release notes

## Questions?

- Open an issue for questions
- Join our community discussions
- Contact maintainers

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Recognition

Contributors are recognized in:
- GitHub contributors page
- CHANGELOG.md
- README.md (for significant contributions)

Thank you for contributing to Wan2.1!
