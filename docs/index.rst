Wan2.1 Documentation
====================

Welcome to the Wan2.1 documentation! Wan2.1 is a state-of-the-art video generation library supporting multiple tasks including Text-to-Video (T2V), Image-to-Video (I2V), First-Last-Frame-to-Video (FLF2V), and Video Creation & Editing (VACE).

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/pipelines
   user_guide/models
   user_guide/configuration
   user_guide/distributed

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/pipelines
   api/utils
   api/distributed

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Quick Links
===========

- `GitHub Repository <https://github.com/Kuaishou/Wan2.1>`_
- `Issue Tracker <https://github.com/Kuaishou/Wan2.1/issues>`_
- `PyPI Package <https://pypi.org/project/wan/>`_

Features
========

Core Capabilities
-----------------

* **Multiple Generation Modes:**

  - Text-to-Video (T2V)
  - Image-to-Video (I2V)
  - First-Last-Frame-to-Video (FLF2V)
  - Video Creation & Editing (VACE)
  - Text-to-Image (T2I)

* **Model Sizes:**

  - 14B parameters (state-of-the-art quality)
  - 1.3B parameters (efficient deployment)

* **Advanced Features:**

  - Flash Attention 2/3 support
  - Distributed training with FSDP
  - Context parallelism (Ulysses/Ring)
  - Prompt extension with LLMs
  - Custom 3D Causal VAE

* **Production Ready:**

  - Single-GPU and multi-GPU support
  - Gradio web interface
  - Diffusers integration
  - Comprehensive testing

System Requirements
===================

Minimum Requirements
--------------------

- Python 3.10+
- PyTorch 2.4.0+
- CUDA 11.8+ (for GPU support)
- 24GB+ GPU memory (for 1.3B model)
- 80GB+ GPU memory (for 14B model)

Recommended
-----------

- Python 3.11
- PyTorch 2.4.1
- CUDA 12.1
- NVIDIA A100 (80GB) or H100

Quick Start
===========

Installation
------------

.. code-block:: bash

   pip install wan

Basic Usage
-----------

.. code-block:: python

   from wan.text2video import WanT2V

   # Initialize pipeline
   pipeline = WanT2V(
       model_path='path/to/model',
       vae_path='path/to/vae',
       device='cuda'
   )

   # Generate video
   video = pipeline(
       prompt="A beautiful sunset over the ocean",
       num_frames=16,
       height=512,
       width=512
   )

License
=======

Wan2.1 is released under the Apache 2.0 License. See the LICENSE file for details.

Citation
========

If you use Wan2.1 in your research, please cite:

.. code-block:: bibtex

   @software{wan2024,
     title={Wan2.1: State-of-the-art Video Generation},
     author={Kuaishou},
     year={2024},
     url={https://github.com/Kuaishou/Wan2.1}
   }
