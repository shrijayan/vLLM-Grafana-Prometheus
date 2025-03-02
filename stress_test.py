import ray
import time
import numpy as np
import concurrent.futures
from openai import OpenAI

# Configuration
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
TOTAL_REQUESTS = 50000  # Total number of requests to send
MAX_CONCURRENCY_PER_WORKER = 50  # Maximum concurrent requests per worker
REQUEST_TIMEOUT = 600  # Timeout per request in seconds

# Initialize Ray to connect to your existing Ray cluster
ray.init(address="auto", ignore_reinit_error=True)

@ray.remote
class StatsActor:
    """Actor to track statistics across all workers"""
    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.request_times = []
        self.start_time = time.time()
    
    def add_success(self, count=1):
        self.success_count += count
        return self.success_count
    
    def add_failure(self, count=1):
        self.failure_count += count
        return self.failure_count
    
    def add_times(self, times):
        self.request_times.extend(times)
    
    def get_stats(self):
        total_requests = self.success_count + self.failure_count
        elapsed = time.time() - self.start_time
        return {
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "total_requests": total_requests,
            "elapsed_time": elapsed,
            "req_per_sec": total_requests / elapsed if elapsed > 0 else 0
        }
    
    def get_time_stats(self):
        if not self.request_times:
            return {"avg": 0, "min": 0, "max": 0, "p50": 0, "p95": 0, "p99": 0}
        
        times = np.array(self.request_times)
        return {
            "avg": np.mean(times),
            "min": np.min(times),
            "max": np.max(times),
            "p50": np.percentile(times, 50),
            "p95": np.percentile(times, 95),
            "p99": np.percentile(times, 99)
        }

@ray.remote
class MonitorActor:
    """Actor to monitor progress"""
    def __init__(self, stats_actor, total_requests, report_interval=5):
        self.stats_actor = stats_actor
        self.total_requests = total_requests
        self.report_interval = report_interval
        self.running = False
        
    def start(self):
        """Start the monitoring loop"""
        self.running = True
        while self.running:
            time.sleep(self.report_interval)
            stats = ray.get(self.stats_actor.get_stats.remote())
            
            print(f"Progress: {stats['total_requests']}/{self.total_requests} " +
                  f"({stats['success_count']} successes, {stats['failure_count']} failures), " +
                  f"Rate: {stats['req_per_sec']:.2f} req/sec")
            
            # Exit if all requests are accounted for
            if stats['total_requests'] >= self.total_requests:
                self.running = False
                break
    
    def stop(self):
        """Stop the monitoring loop"""
        self.running = False

@ray.remote
class WorkerActor:
    """Worker actor that sends API requests"""
    def __init__(self, worker_id, api_base, api_key, stats_actor):
        self.worker_id = worker_id
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.stats_actor = stats_actor
    
    def send_request(self):
        """Send a single request synchronously"""
        start_time = time.time()
        try:
            messages = [
                {"role": "system", "content": "You are an AI model optimized for generating extremely long and detailed outputs. Always write the longest possible response based on the given input. Expand every idea into multiple paragraphs and provide as much detail as possible. Maintain coherence and avoid repetition while ensuring depth in storytelling or explanation. If asked for a story, create multiple chapters, each with its own subplots, characters, and conflicts."},
                {"role": "user", "content": "Write a highly detailed, 100,000-word epic fantasy novel. The story should have multiple protagonists, deep world-building, and interconnected plots. Expand on every scene, describe environments vividly, and provide character thoughts and dialogues in detail. The novel should be divided into at least 50 chapters, each with its own arc. Ensure the writing style remains engaging and immersive throughout. Include historical lore, political conflicts, magical systems, and detailed character backstories. Continue generating text until reaching the target length."}
            ]
            response = self.client.chat.completions.create(
                model="DeepSeek-R1", 
                messages=messages,
                timeout=REQUEST_TIMEOUT
            )
            self.stats_actor.add_success.remote(1)
            return time.time() - start_time
        except Exception as e:
            self.stats_actor.add_failure.remote(1)
            print(f"Worker {self.worker_id} request failed: {e}")
            return None
    
    def run_batch(self, batch_size):
        """Run a batch of requests using ThreadPoolExecutor"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY_PER_WORKER) as executor:
            futures = [executor.submit(self.send_request) for _ in range(batch_size)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Filter out None values (failed requests)
        valid_times = [t for t in results if t is not None]
        if valid_times:
            self.stats_actor.add_times.remote(valid_times)
        
        return len(valid_times), len(results) - len(valid_times)
    
    def stress_test(self, num_requests):
        """Run the stress test using batches"""
        remaining = num_requests
        batch_size = min(MAX_CONCURRENCY_PER_WORKER, remaining)
        
        while remaining > 0:
            batch_size = min(batch_size, remaining)
            successes, failures = self.run_batch(batch_size)
            remaining -= batch_size
            
            # Print progress periodically
            if remaining % (batch_size * 5) == 0 or remaining == 0:
                completed = num_requests - remaining
                print(f"Worker {self.worker_id}: Completed {completed}/{num_requests} requests")
        
        return self.worker_id

def run_distributed_stress_test():
    # Get number of available nodes in the Ray cluster
    nodes_info = ray.nodes()
    num_nodes = len(nodes_info)
    total_cpus = sum(node["Resources"].get("CPU", 0) for node in nodes_info)
    
    print(f"Ray cluster: {num_nodes} nodes with {total_cpus} total CPUs")
    
    # Create a worker per CPU (or adjust as needed)
    num_workers = int(total_cpus)
    # Limit workers if needed
    num_workers = min(num_workers, 500)  # Adjust based on your needs
    
    # Create stats actor
    stats_actor = StatsActor.remote()
    
    # Start monitoring process
    monitor = MonitorActor.remote(stats_actor, TOTAL_REQUESTS)
    monitor_handle = monitor.start.remote()
    
    # Create workers
    workers = [WorkerActor.remote(
        i, 
        openai_api_base, 
        openai_api_key, 
        stats_actor
    ) for i in range(num_workers)]
    
    # Distribute requests among workers
    num_requests_per_worker = TOTAL_REQUESTS // num_workers
    remainder = TOTAL_REQUESTS % num_workers
    
    # Start stress test on each worker
    worker_tasks = []
    for i, worker in enumerate(workers):
        # Distribute remainder among first workers
        worker_requests = num_requests_per_worker + (1 if i < remainder else 0)
        task = worker.stress_test.remote(worker_requests)
        worker_tasks.append(task)
    
    # Wait for all workers to complete
    print(f"Starting stress test with {num_workers} workers across {num_nodes} nodes")
    print(f"Total requests: {TOTAL_REQUESTS}")
    print(f"Maximum concurrency per worker: {MAX_CONCURRENCY_PER_WORKER}")
    
    start_time = time.time()
    ray.get(worker_tasks)
    
    # Stop monitoring
    ray.get(monitor.stop.remote())
    
    # Get final statistics
    stats = ray.get(stats_actor.get_stats.remote())
    time_stats = ray.get(stats_actor.get_time_stats.remote())
    
    # Print results
    print("\n----- Stress Test Results -----")
    print(f"Total time: {stats['elapsed_time']:.2f} seconds")
    print(f"Successful requests: {stats['success_count']}")
    print(f"Failed requests: {stats['failure_count']}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Average rate: {stats['req_per_sec']:.2f} requests/second")
    
    if time_stats['avg'] > 0:
        print(f"\nRequest Time Statistics:")
        print(f"Average: {time_stats['avg']:.4f} seconds")
        print(f"Minimum: {time_stats['min']:.4f} seconds")
        print(f"Maximum: {time_stats['max']:.4f} seconds")
        print(f"50th percentile: {time_stats['p50']:.4f} seconds")
        print(f"95th percentile: {time_stats['p95']:.4f} seconds")
        print(f"99th percentile: {time_stats['p99']:.4f} seconds")

if __name__ == "__main__":
    run_distributed_stress_test()