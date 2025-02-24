import multiprocessing
import asyncio
import torch
from openai import OpenAI

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
TOTAL_REQUESTS = 5000  # Total number of requests to send
CONCURRENCY = 500     # Fixed concurrency value
CORES = multiprocessing.cpu_count()  # Number of CPU cores

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

async def send_request():
    try:
        messages = [
            {"role": "system", "content": "Write as big as possible."},
            {"role": "user", "content": "Write a very very very 100000 words story."}
        ]
        response = client.chat.completions.create(model="DeepSeek-R1", messages=messages)
    except Exception as e:
        print(f"Request failed: {e}")

async def stress_test_process(num_requests):
    tasks = [send_request() for _ in range(num_requests)]
    await asyncio.gather(*tasks)

def run_process(num_requests):
    asyncio.run(stress_test_process(num_requests))

if __name__ == "__main__":
    print(f"Running stress test with concurrency: {CONCURRENCY}")
    num_requests_per_process = TOTAL_REQUESTS // CORES
    processes = []
    
    for _ in range(CORES):
        p = multiprocessing.Process(target=run_process, args=(num_requests_per_process,))
        processes.append(p)
        p.start()
    
    for p in processes:
        p.join()
