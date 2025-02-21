import asyncio
import torch
from openai import OpenAI

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
TOTAL_REQUESTS = 500  # Total number of requests to send
CONCURRENCY = 100     # Fixed concurrency value

client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

# Global counters
processed_count = 0
active_count = 0
counter_lock = asyncio.Lock()

async def send_request(semaphore):
    global processed_count, active_count
    async with semaphore:
        # Increment active count
        async with counter_lock:
            active_count += 1
            current_active = active_count
        print(f"Active concurrent requests: {current_active}")

        try:
            messages = [
                {"role": "system", "content": "When people ask you to write code, follow Uncle BOB's clean code practice."},
                {"role": "user", "content": "Write a very very very 10000 words story."}
            ]
            response = client.chat.completions.create(model="DeepSeek-R1-Distill-Qwen-7B", messages=messages)
        except Exception as e:
            print(f"Request failed: {e}")
        finally:
            async with counter_lock:
                active_count -= 1
                processed_count += 1
                print(f"Processed {processed_count}/{TOTAL_REQUESTS} requests")

async def stress_test():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    tasks = [send_request(semaphore) for _ in range(TOTAL_REQUESTS)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    print(f"Running stress test with concurrency: {CONCURRENCY}")
    asyncio.run(stress_test())
