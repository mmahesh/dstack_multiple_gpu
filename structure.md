# Structure of the blog post 
TITLE: Building a pipeline for training on multiple GPUs with dstack
- Introduction
  - Objectives
      - to create a simple deep learning model to train on the popular MNIST dataset using Pytorch Lightning,
      - to create a dstack workflow for using multiple GPUs,
      - to train the deep learning model on AWS GPUs using dstack workflow.
  - Pre-requisites
    - familiarity with deep learning,
    - familiarity with python,
    - familiarity with pytorch and pytorch-lightning .
- Preliminaries
  - Requirements
    - dstack
    - pytorch-lightning
    - torch
    - torch-vision
  - Directory setup
    ```
    dstack_test/
        .dstack/
            workflows.yaml
            variables.yaml
        train.py
        requirements.txt
    ```

- Deep Learning Model
  - Briefly explain the problem (classification/regression/autoencoder)
  - Briefly explain the data and the model (TODO: do not go in detail).
    - MNIST dataset
    - Autoencoder
  - Explain how the multi GPU setting differs from the standard one GPU setting.
  - In the end mention that in principle one can use a different deep learning model.
- Our Dstack Workflow
  - Explain briefly what dstack is. Briefly explain the benefits of dstack compared to launching an AWS instance by oneself.
  - Explain the dstack setup.
  - Explain the dstack workflow and variables.
  - Finally, show how to run the dstack workflow, monitor the results and GPU utilization.
- Conclusion
  - Explain that in the blog we have seen how to
      - create a deep learning model using Pytorch Lightning suitable for multi-GPU setting,
      - create appropriate dstack workflow for this setting,
      - run the dstack workflow using AWS GPUs,
      - and monitor the results via dstack logs.
- References
  - [Dstack documentation](https://docs.dstack.ai)
  - [Pytorch Lightning](https://www.pytorchlightning.ai/)

