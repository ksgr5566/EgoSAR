import cv2
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
import warnings
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input_folder", type=str, required=True)
parser.add_argument("--output_folder", type=str, required=True)

args = parser.parse_args()

def calculate_optical_flow(frame1, frame2, prev_path, curr_path):
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # Compute magnitude and angle
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    denom = np.max(mag) - np.min(mag)
    with warnings.catch_warnings(record=True) as w:
      if denom == 0:
        print()
        print("-------denom 0------")
        print("prev_path:", prev_path)
        print("curr_path:", curr_path)
        mag = np.zeros_like(mag)
      else:
        mag = (mag - np.min(mag)) / denom
        if len(w) > 0:
          print()
          print("-------Warning-----")
          print("prev_path:", prev_path)
          print("curr_path:", curr_path)
          print(denom)
          print("Pre mag")
          print(mag)
          mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)
          print("Post mag")
          print(mag)
          print("-------------------")
    ang = ang / 2*np.pi
    return mag, ang

def process_subfolder(root):
    files = os.listdir(root)
    if len(files) == 16:  # Found a directory with 16 frames
        output_subfolder = os.path.join(args.output_folder, os.path.relpath(root, args.input_folder))

        # Sort files to make sure they're in the correct order
        files = sorted(files)

        # Initialize an empty array to hold the output data
        output_data = np.zeros((15, 2, 229, 229))

        prev_path = os.path.join(root, files[0])
        prev_frame = cv2.imread(prev_path)
        prev_frame = cv2.resize(prev_frame, (229, 229))  # Resize image

        for i in range(1, 16):
            curr_path = os.path.join(root, files[i])
            curr_frame = cv2.imread(curr_path)
            curr_frame = cv2.resize(curr_frame, (229, 229))  # Resize image

            mag, ang = calculate_optical_flow(prev_frame, curr_frame, prev_path, curr_path)

            output_data[i-1, 0, :, :] = mag
            output_data[i-1, 1, :, :] = ang

            prev_frame = curr_frame  # Update the previous frame
            prev_path = curr_path

        # Save output data to a numpy file
        output_data = output_data.astype(np.float32)
        np.savez_compressed(f"{output_subfolder}.npz", data=output_data)
    else:
      print("Error!")


if __name__ == "__main__":
   num_processes = os.cpu_count()  # Adjust according to your CPU
   subfolders = [f"{args.input_folder}/{item}" for item in os.listdir(args.input_folder)]
   with Pool(num_processes) as p:
      list(tqdm(p.imap_unordered(process_subfolder, args.input_folder, subfolders), total=len(subfolders)))