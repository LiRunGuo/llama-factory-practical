import os
import transformers
import torch


def main():
    # 将此路径替换为你的本地模型路径或 HF 模型名
    model_id = os.getenv("HF_MODEL_ID", "Qwen/Qwen2-1.5B-Instruct")
    pipe = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    messages = [
        {"role": "system", "content": "你是一个电商客服，专业回答售后问题"},
        {"role": "user", "content": "你们这儿包邮吗?"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    terminators = [pipe.tokenizer.eos_token_id]
    outputs = pipe(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    print(outputs[0]["generated_text"][len(prompt):])


if __name__ == "__main__":
    main()


