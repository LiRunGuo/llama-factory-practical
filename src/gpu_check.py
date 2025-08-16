import torch


def main():
    print("PyTorch:", torch.__version__)
    has_cuda = torch.cuda.is_available()
    print("CUDA available:", has_cuda)
    if has_cuda:
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()


