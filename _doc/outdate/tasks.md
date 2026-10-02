# Backlog: Search and Ranking on OTTO

The work runs in two halves. Phases A to F are notebooks only: as a full-stack data scientist you build and understand every algorithm using local tools, with no AWS, no MLflow and no pipelines. Phase F ends with a decision on which models to keep. Only after that do the MLOps levels begin (Phases G to I), where AWS, MLflow and the rest first appear. Phase J is the frontend and wrap-up.

The levels follow these definitions:
- Level 0, manual: notebook-driven training, artifacts handed off by hand, manual and infrequent deployment, no monitoring, no separation between experiment and production code.
- Level 1, pipeline automation: an automated training pipeline, a feature store shared by training and serving, experiment tracking, a model registry, continuous training on new data, and a human approval before production.
- Level 2, CI/CD: pipeline code under CI/CD, automated model validation, automated canary deployment, monitoring with alerting and retraining triggers, comparison of models on live traffic, and full reproducibility from data version to deployed model.

