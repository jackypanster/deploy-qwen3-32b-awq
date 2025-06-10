import time
import random
from locust import HttpUser, task, between

# Replace with your vLLM server's IP if not running locust on the same machine
VLLM_SERVER_IP = "10.49.121.127" # Or "localhost" if running locally
VLLM_SERVER_PORT = 8000

class LLMUser(HttpUser):
    wait_time = between(1, 5)  # Simulate users waiting 1-5 seconds between tasks
    host = f"http://{VLLM_SERVER_IP}:{VLLM_SERVER_PORT}"

    # Sample prompts - expand with more diverse and realistic examples
    sample_prompts = [
        "你好，请介绍一下你自己。",
        "写一首关于春天的五言绝句。",
        "Explain the concept of black holes in simple terms.",
        "Translate 'Hello, world!' to French.",
        "What is the capital of Japan and what is its population?"
    ]

    @task
    def chat_completion(self):
        headers = {
            "Content-Type": "application/json",
            # vLLM typically doesn't require an API key by default
            # "Authorization": "Bearer YOUR_DUMMY_API_KEY"
        }
        
        user_prompt = random.choice(self.sample_prompts)
        max_tokens = random.randint(50, 250) # Vary output length

        payload = {
            "model": "coder",  # Should match --served-model-name
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": max_tokens,
            # "stream": False # Set to True if you want to test streaming
        }

        with self.client.post("/v1/chat/completions", json=payload, headers=headers, catch_response=True) as response:
            if response.ok:
                try:
                    response_data = response.json()
                    # You could add more detailed checks or custom metrics here
                    # For example, counting tokens in response:
                    # output_tokens = len(response_data['choices'][0]['message']['content'].split()) # Naive token count
                    # response.success() # Already successful if response.ok
                except Exception as e:
                    response.failure(f"Failed to parse JSON or process response: {e}")
            else:
                response.failure(f"Request failed with status {response.status_code}: {response.text}")

    # You can add more tasks to simulate different API calls or scenarios
    # @task(2) # Example: make this task twice as likely
    # def another_task(self):
    #     pass
