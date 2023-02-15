import argparse
import os
from pcd_editing import write_pth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--ckpt", help = "Provide the path of the pth file", type=str)
    parser.add_argument("--ckpt_deformed", help = "Provide the path of the pth file", type=str)
    parser.add_argument("--pcd", default='', help = "Provide the path of the ply file to be saved", type=str)
    
    args = parser.parse_args()

    pth_file = os.path.join(args.save_dir, args.ckpt)
    pth_file_deformed = os.path.join(args.save_dir, args.ckpt_deformed)
    pcd_file = os.path.join(args.save_dir, args.pcd)

    write_pth(pth_file, pth_file_deformed, pcd_file)
    

if __name__ == "__main__":
    main()