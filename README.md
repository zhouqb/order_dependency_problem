# Explore Order Dependency Problema

## Introduction
Answering Multiple Choice Questions (MCQs) is a common task for Large Language Models (LLMs), frequently used in benchmarks to assess their performance as well as in automatic evaluation frameworks. Ideally, LLMs should consistently select accurate answers based solely on the question's content and the available options. However, multiple studies have shown that LLMs are susceptible to changes in the order of answer choices, known as the Order Dependency Problem (ODP). This repo implements various methods to explore the order dependency problem.

## Get Started
1. Use `conda`/`miniconda` to create an environment
```bash
conda create --name order_dependency_problem python
conda activate order_dependency_problem
```

2. Install the package
```bash
cd order_dependency_problem/
pip install -e .
```

3. Follow the example notebook `notebooks/explore_order_dependency_problem.ipynb`. We also have sample data from ARC and MMLU in `data` directory.

## Metrics
We implemented three types of metrics to quantify ODP.
* Answer prevalency. This is a biased metric for ODP, since it is sensitive to ground truth prevalence.
* Answer accuracy. Accuracy fluctuation under "answer-moving attack" (moving all ground truths to a specific position/option) is an indicator of ODP.
* Standard deviation of recall balance. This is a direct quantitative measure of ODP. Since it is based on the recall per ground truth label, it is not sensitive to ground truth prevalence.

## Experiments
We explored the following experiments.
* Answer-moving attack
* Shuffling option contents.
* Shuffling option IDs.
* Removing option IDs.

