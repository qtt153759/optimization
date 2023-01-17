from library.utils.load_data import Load
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--dataset", type=str)
    param = parser.parse_args()
    
    data = Load()
    data(param.dataset)
    print("\n\nINFO - DATA:")
    print(data)