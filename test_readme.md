Region: Ohio - us-east-2b
Name: TW_India_llm_experiment
OS: Ubuntu 
Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6 (Ubuntu 22.04)
Instance Type: 
g4dn.xlarge, g6.12xlarge, p4d.24xlarge, p5en.48xlarge
Key Pair: tw_india_shrijayan
Change to Public Subnet
Network: SSH My IP
Storage: 100
My IP: 14.142.51.155/32


Security Group Edit: 
```
Custom TCP
0-65535
shrijayan-security-group
```
![Security Group](image-1.png)

<!-- Initial Setup:
```
sudo apt update
sudo apt install python3.12-dev
sudo apt install -y build-essential libglvnd-dev pkg-config
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

sudo apt install -y nvidia-driver-570
sudo apt install nvidia-utils-570
sudo apt install -y nvidia-fabricmanager-570
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager
sudo apt install nvidia-cuda-toolkit

sudo reboot

nvidia-smi
``` -->

```
sudo mkdir -p /mnt
sudo lsblk
```
```
sudo mount /dev/nvme /mnt
```

Tmux:
```
sudo apt install tmux
tmux new -s vllm
```

Installing Dependencies:
```
python3.12 -m venv vllm
source vllm/bin/activate
pip install vllm ray python-etcd
```

```
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_P2P_DISABLE=0
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_BUFFSIZE=4194304
export NCCL_SOCKET_IFNAME=ens32
export NCCL_DEBUG=INFO
export GLOO_DEBUG=2
export VLLM_LOGGING_LEVEL=DEBUG
export GLOO_SOCKET_IFNAME=ens32
export NCCL_ASYNC_ERROR_HANDLING=1
export RAY_memory_monitor_refresh_ms=0
```


Ray Connect:
```
ray start --head --port=6379 --dashboard-host=0.0.0.0
```
Take the ID and paste in other machine to connect to the same ray cluster

```
ray status
```

Git clone the repository:
```
git clone https://github.com/shrijayan/vLLM-Grafana-Prometheus.git
cd vLLM-Grafana-Prometheus
sudo docker compose up -d
```

### Change Prometheus.yml file IP address to the private IP of the machine

Machine 1
```
torchrun --nnodes=2 --node_rank=0 --master_addr=172.31.20.216 --master_port=29500 --nproc-per-node=8 multiprocess_test.py
```

Machine 2
```
torchrun --nnodes=2 --node_rank=1 --master_addr=172.31.20.216 --master_port=29500 --nproc-per-node=8 multiprocess_test.py
```

<!-- Docker Install:
```
sudo apt update
sudo apt install apt-transport-https ca-certificates curl software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
apt-cache policy docker-ce
sudo apt install docker-ce
sudo systemctl status docker
``` -->

# Copy the Private IP and paste in thr prometheus.yml file



Grafana > Connections > Data Sources > Prometheus
```
URL: http://<AWS_URL>:9090
```

Grafana
Copy Paste for dashboard:
```
https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/prometheus_grafana/grafana.json 
```

<!-- ```
export NCCL_DEBUG=INFO
export GLOO_DEGUB=2
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_SOCKET_IFRAME=wlo1
export GLOO_SOCKET_IFRAME=wlo1
export NCCL_ASYNC_ERROR_HANDLING=1
export RAY_memory_monitor_refersh_ms=0
``` -->

<!-- Install Model:
```
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
``` -->

vLLM Serve:
```
python -m vllm.entrypoints.openai.api_server \
--model /mnt/home/ubuntu/DeepSeek-R1-Distill-Llama-70B \
--served-model-name DeepSeek-R1 \
--enable-reasoning \
--reasoning-parser deepseek_r1 \
--dtype float16 \
--port 8000 \
--gpu_memory-utilization 0.99 \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2 \
--trust-remote-code

```

```
python -m vllm.entrypoints.openai.api_server \
--model /mnt/home/ubuntu/DeepSeek-R1 \
--served-model-name DeepSeek-R1 \
--enable-reasoning \
--reasoning-parser deepseek_r1 \
--dtype float16 \
--port 8000 \
--gpu_memory-utilization 0.99 \
--tensor-parallel-size 8 \
--pipeline-parallel-size 6 \
--trust-remote-code

```

cURL:
```
curl http://0.0.0.0:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "DeepSeek-R1",
        "messages": [{"role": "user", "content": "San Francisco is a"}],
        "temperature": 0
    }'
```












Maybe 
![alt text](image.png)


How to download the model?
S3 or image replicate or volume replicate

How to create volume replicate?

How to download the model in low configure machine?
4GB ram not enough 32GB ram mahcine needed

Who to create the EC2 with DeepLearning Image?
Deep Learning AMI (Ubuntu 22.04) Version 42.0

How to two ec2 mahcine in HPC?

Why it is slow when I rent 10 24GB GPU machine?