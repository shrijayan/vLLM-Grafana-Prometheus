Region: N. Virginia - us-east-1
Name: TW_India_llm_experiment
OS: Ubuntu
Instance Type: g4dn.xlarge
Key Pair: tw_india_shrijayan
Network: SSH My IP
Storage: 100

Security Group Edit: 
```
Custom TCP
0-65535
shrijayan-security-group
```

Initial Setup:
```
sudo apt update
sudo apt install -y build-essential libglvnd-dev pkg-config
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

sudo apt install -y nvidia-driver-570

sudo apt install -y nvidia-fabricmanager-570
sudo systemctl enable nvidia-fabricmanager
sudo systemctl start nvidia-fabricmanager

sudo reboot

nvidia-smi

```

Installing Dependencies:
```
sudo apt install python3.12-venv
python3.12 -m venv vllm
source vllm/bin/activate

pip install vllm ray
```

Ray Connect:
```
ray start --head --port=6379
```
Take the ID and paste in other machine to connect to the same ray cluster

```
ray status
```

Tmux:
```
sudo apt install tmux
tmux new -s vllm
```

Install Model:
```
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

