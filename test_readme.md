Region: N. Virginia - us-east-1
Name: TW_India_llm_experiment
OS: Ubuntu 
Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.5 (Ubuntu 22.04)
Instance Type: 
g4dn.xlarge
g6.12xlarge
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

sudo mount /dev/nvme1n1p1 /mnt
```

Tmux:
```
sudo apt install tmux
tmux new -s vllm

source vllm/bin/activate
```

Installing Dependencies:
```
sudo apt install python3.12-venv
python3.12 -m venv vllm
source vllm/bin/activate
pip install vllm ray
```

```
export NCCL_DEBUG=INFO
export GLOO_DEBUG=2
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_SOCKET_IFNAME=enp3s0
export GLOO_SOCKET_IFNAME=enp3s0
export NCCL_ASYNC_ERROR_HANDLING=1
export RAY_memory_monitor_refresh_ms=0
```


Ray Connect:
```
ray start --head --port=6379
```
Take the ID and paste in other machine to connect to the same ray cluster

```
ray status
```
## Ray Test
```
# Test PyTorch NCCL
import torch
import torch.distributed as dist
dist.init_process_group(backend="nccl")
local_rank = dist.get_rank() % torch.cuda.device_count()
torch.cuda.set_device(local_rank)
data = torch.FloatTensor([1,] * 128).to("cuda")
dist.all_reduce(data, op=dist.ReduceOp.SUM)
torch.cuda.synchronize()
value = data.mean().item()
world_size = dist.get_world_size()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch NCCL is successful!")

# Test PyTorch GLOO
gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
cpu_data = torch.FloatTensor([1,] * 128)
dist.all_reduce(cpu_data, op=dist.ReduceOp.SUM, group=gloo_group)
value = cpu_data.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("PyTorch GLOO is successful!")

if world_size <= 1:
    exit()

# Test vLLM NCCL, with cuda graph
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

pynccl = PyNcclCommunicator(group=gloo_group, device=local_rank)
# pynccl is enabled by default for 0.6.5+,
# but for 0.6.4 and below, we need to enable it manually.
# keep the code for backward compatibility when because people
# prefer to read the latest documentation.
pynccl.disabled = False

s = torch.cuda.Stream()
with torch.cuda.stream(s):
    data.fill_(1)
    out = pynccl.all_reduce(data, stream=s)
    value = out.mean().item()
    assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL is successful!")

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g, stream=s):
    out = pynccl.all_reduce(data, stream=torch.cuda.current_stream())

data.fill_(1)
g.replay()
torch.cuda.current_stream().synchronize()
value = out.mean().item()
assert value == world_size, f"Expected {world_size}, got {value}"

print("vLLM NCCL with cuda graph is successful!")

dist.destroy_process_group(gloo_group)
dist.destroy_process_group()
```

```
NCCL_DEBUG=TRACE torchrun --nnodes 2 --nproc-per-node=2 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR test.py
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

Git clone the repository:
```
git clone https://github.com/shrijayan/vLLM-Grafana-Prometheus.git
cd vLLM-Grafana-Prometheus
sudo docker compose up -d
```

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

python -m vllm.entrypoints.openai.api_server \
--model /mnt/home/ubuntu/DeepSeek-R1-Distill-Llama-70B \
--served-model-name DeepSeek-R1 \
--enable-reasoning \
--reasoning-parser deepseek_r1 \
--dtype float \
--port 8000 \
--gpu_memory-utilization 0.99 \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2


vLLM Serve:
```
python -m vllm.entrypoints.openai.api_server \
--model /home/ubuntu/DeepSeek-R1-Distill-Qwen-7B \
--served-model-name DeepSeek-R1 \
--enable-reasoning \
--reasoning-parser deepseek_r1 \
--dtype float16 \
--port 8000 \
--gpu_memory-utilization 0.98 \
--tensor-parallel-size 8 \
--pipeline-parallel-size 2

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