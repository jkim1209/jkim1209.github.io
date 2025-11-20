---
tags: Python, MLOps, FastAPI, Docker, Airflow, React, Monitoring
date: 2025
icon: 🎬
---

# MLOps Project: Movie Recommendation System

## Project Overview

This project builds an MLOps pipeline managing the entire lifecycle of machine learning models, from data collection to model training, serving, and monitoring.

GitHub Repository: [https://github.com/jkim1209/mlops-project](https://github.com/jkim1209/mlops-project)

Presentation Slides: [Google Drive](https://drive.google.com/file/d/125RqKVFW9l_Nk6yo8OS24BwmA_vdXj8Z)

## My Role and Contributions

Oversaw the machine learning and backend aspects of the project, playing a key role from model development to API serving and frontend integration.

- Modeling & Experiment Management: Implemented a lightweight Numpy-based MLP recommendation model and systematically tracked and managed all experimental processes and results using MLflow.
- API Server Development: Built an API server providing real-time model inference results using FastAPI. Designed and implemented various endpoints including `/predict` and `/latest-recommendations`.
- System Deployment: Integrated the model-serving API with a React-developed frontend to enable actual user service access, and built the deployment environment using Docker.

## Key Technologies and Implementation

### Technologies Used

- Python
- ML & Data Science: Numpy, Pandas, Scikit-learn
- MLOps & Backend: FastAPI, PostgreSQL, AWS S3, Docker, Airflow
- Frontend: React.js
- Monitoring: MLflow, Prometheus, Grafana, Loki

### Core Implementation

- Implemented Numpy-based MLP recommendation model
- Model training monitoring using MLflow
- FastAPI-based real-time model inference API server
- Automated data-training-inference pipeline using Airflow DAG
- Built real-time server and application monitoring system using Prometheus, Grafana, and Loki

### MLOps Architecture

![MLOps Architecture](/projects/assets/images/02/01.png)

## Troubleshooting

### Problem: Recommendation Result Loss Due to Database Loading Failure

**Problem Description**

Recommended movie posters and titles weren't displaying in the React frontend. Investigation revealed that user viewing records weren't being stored in the PostgreSQL database at all.

**Solution**

1. Debugging the data loading logic revealed that when the **`release_date` field was an empty string ("")**, SQL queries generated errors, causing the entire data insertion to fail.
2. **Cause of Detection Delay**:. Cause of Detection Delay: The model training/inference pipeline referenced locally cached log files instead of the DB, making it appear to function normally. This prevented detection of data loading errors for some time.
3. Modified data loading code to **handle exceptions when `release_date` field is an empty string (e.g., converting to NULL)**, resolving the issue.

![Monitoring Dashboard](/projects/assets/images/02/02.png)

## Results and Achievements

Successfully built and deployed a complete MLOps pipeline including data collection, model training, batch inference, API serving, and monitoring.

### Learnings and Insights

- Gained development experience organically connecting individual technologies (Docker, FastAPI, Airflow, etc.) to create one complete service.
- Experienced firsthand how silent failures in one part of a pipeline can affect the entire system, deeply appreciating the importance of end-to-end testing and robust logging systems.
- Learned the value of collaborative communication by working with team members in unfamiliar fields, understanding and coordinating each other's expertise.

![Final Results](/projects/assets/images/02/03.png)
