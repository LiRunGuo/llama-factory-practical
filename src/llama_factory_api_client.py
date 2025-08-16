import os
from openai import OpenAI
from transformers.utils.versions import require_version


def main():
    require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")
    port = int(os.environ.get("API_PORT", 8000))
    client = OpenAI(api_key="0", base_url=f"http://localhost:{port}/v1")
    messages = [{"role": "user", "content": "hello, where is USA"}]
    result = client.chat.completions.create(messages=messages, model="test")
    print(result.choices[0].message)


if __name__ == "__main__":
    main()


