import argparse
import os
from pcd_editing import get_pcd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ckpt", help = "Provide the path of the pth file", type=str)
    parser.add_argument("--pcd", default='', help = "Provide the path of the ply file to be saved", type=str)
    args = parser.parse_args()

    pth_file = os.path.join(args.save_dir, args.ckpt)
    pcd_file = os.path.join(args.save_dir, args.pcd)

    get_pcd(pth_file, pcd_file)

if __name__ == "__main__":
    main()