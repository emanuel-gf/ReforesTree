from torchgeo.datasets import ReforesTree
import argparse
def main(args):
    pathdata = args.data_dir
    ds = ReforesTree(root=pathdata, download=True, checksum=True)

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Download ReforesTree dataset")
    args.add_argument("--data-dir", type=str,
                       required=True, help="Directory to download the dataset to")
    args = args.parse_args()
    main(args)