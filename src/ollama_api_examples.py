import os
from openai import OpenAI


def main():
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    client = OpenAI(base_url=f"{base_url}/v1/", api_key="ollama")

    # Chat example
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say this is a test"}],
        model="llama3",
    )
    print(chat_completion.choices[0].message)

    # Embeddings example
    embeddings = client.embeddings.create(
        model="all-minilm",
        input=["why is the sky blue?", "why is the grass green?"],
    )
    print("Embeddings dims:", len(embeddings.data[0].embedding))


if __name__ == "__main__":
    main()


