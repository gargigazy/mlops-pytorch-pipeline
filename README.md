# mlops-pytorch-pipeline

**Assignment 03 - Deploying PyTorch ML Workloads with Docker & Kubernetes**

**Name:** Gargi Gupta
**Roll No:** da25m565

This project takes a PyTorch image classifier (ResNet-18 on CIFAR-10) through the full deployment lifecycle - local training, Docker containers, and finally deployment on Kubernetes with training Jobs and a serving Deployment.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │            Kubernetes Cluster                │
                        │         (namespace: ml-training)             │
                        │                                              │
  ┌───────────┐         │  ┌────────────┐        ┌──────────────────┐ │
  │  GitHub   │ push    │  │ ConfigMap  │───────>│   Training Job   │ │
  │   repo    │────────>│  │ (training  │ mount  │  (mlops-train:v1)│ │
  └───────────┘  CI     │  │  config)   │        └────────┬─────────┘ │
                        │  └────────────┘                 │ writes    │
                        │                                 v           │
                        │  ┌───────────┐         ┌────────────────┐   │
                        │  │ data-pvc  │         │ checkpoints-pvc│   │
                        │  └───────────┘         └───────┬────────┘   │
                        │                                │ read-only  │
                        │                                v            │
  ┌───────────┐         │  ┌─────────┐  port 80  ┌───────────────────┐│
  │  client   │────────>│  │ Service │─────────> │ Serving Deployment││
  │ (curl)    │ port-   │  │ClusterIP│  :8080    │ 2x mlops-serve:v1 ││
  └───────────┘ forward │  └─────────┘           │ + liveness/ready  ││
                        │       ^                └───────────────────┘│
                        │       │    ┌─────┐            ^             │
                        │       └────│ HPA │────────────┘             │
                        │            └─────┘  scales 2-5 pods on CPU  │
                        └─────────────────────────────────────────────┘
```

## Project Structure

```
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml    # CI - runs tests + docker builds
├── src/
│   ├── model.py                # resnet18 / simple cnn
│   ├── dataset.py              # cifar10 dataloaders
│   ├── train.py                # training loop with early stopping
│   └── serve.py                # flask app (/predict and /health)
├── configs/training_config.yaml
├── docker/
│   ├── Dockerfile.train        # multi-stage training image
│   └── Dockerfile.serve        # slim serving image (non-root + healthcheck)
├── k8s/                        # all kubernetes manifests
├── requirements/               # pinned deps (train.txt, serve.txt)
└── tests/test_model.py
```

## Setup (local)

```bash
# make a virtual env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/train.txt

# run tests
pip install pytest
pytest tests/ -v

# train locally (downloads cifar10 the first time, takes a while on cpu)
python src/train.py
```

Note: for local training edit `configs/training_config.yaml` and change `data_dir` to `./data` and `checkpoint_dir` to `./checkpoints` (the /app paths are for docker).

## Docker

```bash
# build training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

# run training with mounted volumes
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1

# build serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# run serving
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1

# test it
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Kubernetes (minikube)

```bash
# if using minikube, build the images inside minikube's docker
eval $(minikube docker-env)
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

# apply everything in order
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/training-job.yaml

# watch the training logs
kubectl logs -f job/model-training -n ml-training

# after training finishes deploy the serving layer
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml

# check everything is running
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

# test predictions
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

## Sample prediction response

```json
{
  "predicted_class": "cat",
  "confidence": 0.8123,
  "probabilities": {"airplane": 0.001, "cat": 0.8123, "...": "..."}
}
```

## Git Workflow

- `main` - stable branch
- `develop` - integration branch
- feature branches like `feature/pytorch-model`, `feature/docker-training`, `feature/k8s-deployment` merged via PRs with conventional commit messages (feat:, fix:, docs: etc)
