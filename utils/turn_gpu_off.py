import os


def turn_gpu_off():
    # turn off gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    print(0)