- OS is Ubuntu
- vLLM is running

Assumption:
Acceptance Criteria:
- In Grafana dashboard, developer can able to see the metrics generated from vLLM

Problem Statement:
- Developer needs to see the metrics generated from vLLM in Grafana dashboard

Steps:
Some how you start the vLLM it must be runnning in localhost:8000
```
vllm serve akjindal53244/Llama-3.1-Storm-8B --tensor-parallel-size 2 --host 10.132.3.11 --port 8000 --gpu-memory-utilization 0.95
```
```
docker compose up -d
```
Open Grafana dashboard and Prometheus in browser

http://10.132.3.11:3000/login
Username: admin
Password: admin

click skip

Data Source > Prometheus
URL: http://prometheus:9090
Click Save & Test

Dashboards > Manage > Import
https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/prometheus_grafana/grafana.json

Copy paste the JSON text in the above link in the `Import via dashboard JSON model` field and click Load

Select Prometheus as the data source and click Import
http://10.132.3.11:9090/query

```

1. Docker compose up -d
2. Installing grafana for visualization - http://localhost:3000
Prometheus as Time Series Database
Prometheus scraping job to hit vLLM metrics endpoint and store the event data/metrics in Prometheus db
Setup grafana password: Easy@2025
Connecting grafana inside 
http://localhost:3000
Data Sources
Parameters: Prometheus URL: http://localhost:9090
Click Button: Connect and Test
Import ready made dashboard
Github Link(https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/prometheus_grafana/grafana.json) - Copy paste json content in import dashboard feature in grafana dashboards menu
Use model `akjindal53244/Llama-3.1-Storm-8B`



Solution:
```
vllm serve akjindal53244/Llama-3.1-Storm-8B --tensor-parallel-size 2 --host 10.132.3.11 --port 8000 --gpu-memory-utilization 0.95
```

http://localhost:9090/targets 

https://chatgpt.com/share/67b5c5eb-3f0c-8011-883f-9948129a2195 

https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/prometheus_grafana/grafana.json 

http://localhost:3000/d/b281712d-8bff-41ef-9f3f-71ad43c05e9b/vllm?orgId=1&from=now-5m&to=now&timezone=browser&var-DS_PROMETHEUS=fedj5f5ovlfcwa&var-model_name=akjindal53244%2FLlama-3.1-Storm-8B


### VLLM - Infra setup

- Grafana
- Prometheus
- Jobs to Hit `/metrics` endpoint - scraping


```sh
docker compose up -d
```
