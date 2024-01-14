import pre_alloc_test
import torch_test



def main():
    torch_test.iterating_test()
    pre_alloc_test.pre_alloc_loading_test()


if __name__ == "__main__":
    main()