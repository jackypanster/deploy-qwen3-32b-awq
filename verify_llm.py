import openai

# Configure the OpenAI client to connect to your local vLLM server
client = openai.OpenAI(
    base_url="http://10.49.121.127:8000/v1",
    api_key="dummy-key"  # vLLM doesn't require a real API key by default
)

# Define the messages for the chat completion
# We'll use a simple prompt to test.
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍一下你自己。"}
]

print("Sending request to the LLM...\n")

try:
    # Send the chat completion request
    completion = client.chat.completions.create(
        model="coder",  # This should match the --served-model-name in your Docker command
        messages=messages,
        temperature=0.7,
        max_tokens=150
    )

    # Print the response
    response_content = completion.choices[0].message.content
    print("LLM Response:")
    print(response_content)

    # Verification for enable_thinking=False
    if "<think>" in response_content or "</think>" in response_content:
        print("\nVERIFICATION FAILED: '<think>' tags found in the response.")
    else:
        print("\nVERIFICATION SUCCESSFUL: No '<think>' tags found. 'enable_thinking=False' is working as expected.")

except openai.APIConnectionError as e:
    print(f"Failed to connect to the server: {e}")
    print("Please ensure the vLLM server is running and accessible at http://10.49.121.127:8000.")
except Exception as e:
    print(f"An error occurred: {e}")
