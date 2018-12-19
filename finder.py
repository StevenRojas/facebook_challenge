from app.utils.utils import parse_arguments
from app.network.arch_handler import ArchHandler
import torch
import os


def main():
    args = parse_arguments()
    print(args)


if __name__ == "__main__":
    main()
