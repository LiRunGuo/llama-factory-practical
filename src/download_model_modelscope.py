import os
from dotenv import load_dotenv
from modelscope import snapshot_download


def main():
    load_dotenv()
    local_dir = os.getenv("MODELSCOPE_LOCAL_DIR", os.getcwd())
    # 示例：下载 Qwen2-1.5B-Instruct
    model_id = 'qwen/Qwen2-1.5B-Instruct'
    target_dir = snapshot_download(model_id, local_dir=local_dir)
    print("Downloaded to:", target_dir)


if __name__ == "__main__":
    main()


