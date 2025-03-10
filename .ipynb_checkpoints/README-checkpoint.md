# **Addressing Voter Turnout Disparities through Data-Driven Resource Allocation**
**DSC180 Project by Cici and Angelina**

## **Winter 2025 (WI25)**

Voter participation is a cornerstone of democracy, yet disparities in turnout rates persist across racial and socioeconomic groups. Research from the **Brennan Center** indicates that the racial turnout gap has consistently widened from **2012 to 2022**. While "Get Out The Vote" campaigns have boosted voter participation overall, **inequities in resource allocation** continue to hinder efforts to address these disparities effectively.

Existing research primarily focuses on **predictors of voter turnout** or how racial groups’ **policy preferences impact participation**, but little attention has been devoted to **actionable solutions** for reducing turnout disparities. In the most recent election, **64% voter turnout** was recorded, meaning over **one-third of eligible voters abstained**—disproportionately from **underserved communities**.

### **Key Research Question**
How can resources be strategically reallocated in **Georgia**, particularly in **underserved areas**, to **reduce turnout disparities** and **promote equitable access to voting opportunities**?

### **Project Goals**
This project aims to bridge this gap by leveraging:
- **Causal modeling** to understand the impact of different interventions.
- **Optimization techniques** to design interventions that **maximize voter turnout** while **minimizing disparities**.

---

## **Running the Code**
### **Implementation**
To run the optimization model, execute:

```bash
python WI25/optimize.py
```
or
our_implementation.ipynb

### **Dataset**
The model runs on the `GA_features` dataset.

### **Output Graphs**
Results can be visualized using Our_plot.ipynb

---

## **Fall 2024 (FA24) - Replication Study**
### **Paper 1: Active Learning for Optimal Intervention Design in Causal Models**

This is a replication study of the paper **"Active Learning for Optimal Intervention Design in Causal Models"** (Nature Machine Intelligence, 2023). Specifically, we focused on replicating the experiment on synthetic data.

### **Installation**
Follow these steps to set up the environment:

1. Create a conda environment using `environment.yml` (installation takes ~5 minutes):
   ```bash
   cd paper1
   conda env create -f environment.yml
   ```
2. Install the package in editable mode inside the conda environment:
   ```bash
   pip install -e .
   ```

### **Synthetic Data Experiments**
To run an experiment on synthetic data:

```bash
python run.py --nnodes 5 --noise_level 1 --DAG_type path --std --a_size 2 --a_target 3 4 --acquisition greedy
```

This executes a synthetic data experiment using a specified acquisition function. Results are stored as three pickle files in the output folder.

To visualize the results:
```bash
python visualize_results.py
```
More examples are available in `paper1/optint/notebook/test_multigraphs.ipynb`.

### **Results Storage**
- Results comparison is stored in `paper1/optint/results`.
- Source code is in `paper1/optint/`.

---

## **Paper 2: Reducing Inequality through Optimal Intervention Design**

### **Overview**
This repository contains code and materials for the project **"Reducing Inequality through Optimal Intervention Design."** It integrates two complementary approaches:

#### **1. Active Learning for Optimal Intervention Design in Causal Models**
This approach uses active learning within causal models to efficiently design interventions that shift a system’s mean toward a desired target with minimal intervention effort.

#### **2. Making Decisions that Reduce Discriminatory Impact**
This approach incorporates fairness constraints into causal optimization, ensuring that interventions maximize societal benefit while addressing systemic inequities across protected groups.

### **Repository Contents**
- Implementation of these frameworks.
- Supporting datasets.
- Scripts to reproduce results from the report.

### **Reproducing Results**
To generate figures and print results from the paper:
```bash
python plot_results.ipynb
```

To rerun optimization problems, execute:
```bash
python optimize_disaggregated.py
python optimize.py
```

New `.npy` result files will be output upon re-running. Note that execution may take time.

### **Additional Resources**
- **arXiv link**: [Disaggregated Interventions to Reduce Inequality](https://arxiv.org/abs/2107.00593)
- Documentation and further code restructuring are forthcoming.
