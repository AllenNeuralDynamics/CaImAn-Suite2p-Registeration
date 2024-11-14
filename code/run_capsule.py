import os
import csv
import argparse
import re
import json
import glob
import dateparser
import numpy as np
import logging
from tenacity import retry, stop_after_attempt, wait_fixed
import warnings
from utils.caiman import CaImAnRegistration
from utils.suite2p import suite2pRegistration

def process_file(fn, folder_number, output_path, use_suite2p, use_caiman, writetiff):
    try:
        name, ext = os.path.splitext(os.path.basename(fn))
        success = True  # Assume success unless an error occurs

        if use_suite2p:
            print('fn', fn, folder_number)
            # 1. Suite2p Registration
            suite2p_fn = f"{name}"
            output_path_suite2 = os.path.join(output_path, os.path.join(folder_number, suite2p_fn))
            suite2pRegistration(fn, output_path, folder_number, output_path_suite2)
        else:
            print('-------Skipping Suite2p------')

        if use_caiman:
            # 2. CaImAn Registration
            caiman_fn = f"{name}"
            output_path_caiman = os.path.join(output_path, os.path.join(folder_number, caiman_fn))
            CaImAnRegistration(fn, output_path_caiman)
        else:
            print('-------Skipping CaImAn------')

        return success  # Return True if everything went well

    except Exception as e:
        logging.error(f"Error processing file {fn}: {e}")
        return False  # Return False if an error occurred

def run(data_dir, output_path, use_suite2p, use_caiman, writetiff):
    print('data_dir--->', data_dir)
    if not os.path.exists(output_path):
        print('Creating main output directory...')
        os.makedirs(output_path)
        print('Output directory created at', output_path)

    tif_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tif'):
                file_path = os.path.join(root, file)
                folder_number = os.path.basename(os.path.dirname(file_path))
                tif_files.append((file_path, folder_number))

    print('tif_files--->', tif_files)
    tif_files_sorted = sorted(tif_files, key=lambda x: int(x[1]))

    logging.basicConfig(level=logging.INFO)

    failed_files = []
    
    for i, (fn, folder_number) in enumerate(tif_files_sorted, start=1):
        print(f'Files left: {len(tif_files) - i}')
        if not process_file(fn, folder_number, output_path, use_suite2p, use_caiman, writetiff):
            failed_files.append((fn, folder_number))

    # Retry processing failed files
    if failed_files:
        print("Retrying failed files...")
        for fn, folder_number in failed_files:
            print(f'Retrying file: {fn}')
            process_file(fn, folder_number, output_path, use_suite2p, use_caiman, writetiff)

if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')
    parser.add_argument('--writetiff', type=bool, default=False, help='Save registered tif of the movies')
    parser.add_argument('--suite2p', type=bool, default=False, help='Register movie using suite2p')
    parser.add_argument('--caiman', type=bool, default=False, help='Register movie using caiman')

    # Parse the arguments
    args = parser.parse_args()

    run(args.input, args.output, args.suite2p, args.caiman, args.writetiff)