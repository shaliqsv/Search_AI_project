# 001. Serverless serving architecture (Option A)

Date: 2026-09-19. Status: Accepted. Confirms D26 direction; D26 stays Provisional until the week 22-23 cold-start measurement.

## Decision

Serve from an arm64 Lambda container image behind a Function URL, with the model bundle baked into the image. Neural models run through ONNX Runtime, LightGBM runs natively, FAISS loads at init. No provisioned concurrency, no VPC.

```
Static frontend (S3 + CloudFront / GitHub Pages)
  |  reads versioned JSON exports (default: static replay)
  |  optional live mode (API key)
  v
Lambda Function URL -> Lambda alias (CodeDeploy canary)
  |-> Bedrock (classifier, explainer)
  |-> DynamoDB on-demand (item features)
  |-> in-process: FAISS, ONNX Runtime, LightGBM
CloudWatch alarms -> CodeDeploy auto-rollback
Airflow (local Docker Compose) -> MLflow -> S3 bundle -> ECR image
```

## Why

Traffic will be very low and the goal is minimum cost. Pay-per-request Lambda has zero idle cost and the free tier covers demo traffic. A baked bundle makes every version immutable, so alias = image version and rollback is clean.

## Accepted downsides and mitigations

- Cold starts affect live mode only, since the frontend defaults to static replay.
- Public URL calling Bedrock is a cost-abuse risk: API key, reserved concurrency cap, response cache, budget alarm, kill switch (D32).
- Image size and memory limits: measure early.

## Alternatives considered

- Fargate behind an ALB: no cold starts, but roughly $10-15 a month for the task plus ALB charges. Kept as the fallback.
- Lambda with model pulled from S3 at cold start: smaller images, but breaks immutability and the alias-to-version mapping.

## Revisit trigger

Switch to Fargate only if measured cold start or image size makes live mode unusable (weeks 22-23).
