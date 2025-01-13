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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def process_file(fn, folder_number, output_path, use_suite2p, use_caiman, caiman_intial_temp, jorncorre_intial_temp):
    print('In process file')
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
            CaImAnRegistration(fn, output_path_caiman, caiman_intial_temp, jorncorre_intial_temp, output_shape=None, constant_values=0)

        else:
            print('-------Skipping CaImAn------')

        return success  # Return True if everything went well

    except Exception as e:
        logging.error(f"Error processing file {fn}: {e}")
        return False  # Return False if an error occurred

def run(data_dir, output_path, use_suite2p, use_caiman, writetiff, caiman_intial_temp, jorncorre_intial_temp):
    print('data_dir--->', data_dir)
    if not os.path.exists(output_path):
        print('Creating main output directory...')
        os.makedirs(output_path)
        print('Output directory created at', output_path)

    # Iterate over all files in the directory
    for filename in os.listdir(data_dir):
        # Construct full file path
        file_path = os.path.join(data_dir, filename)
        
        # Check if the file is a .tif file
        if os.path.isfile(file_path) and filename.endswith('.tif'):
            folder_number =  os.path.basename(data_dir)
            print('folder_number', folder_number)
            process_file(file_path, folder_number, output_path, use_suite2p, use_caiman, caiman_intial_temp, jorncorre_intial_temp)

if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')
    parser.add_argument('--writetiff', type=bool, default=False, help='Save registered tif of the movies')
    parser.add_argument('--suite2p', type=bool, default=False, help='Register movie using suite2p')
    parser.add_argument('--caiman', type=bool, default=False, help='Register movie using caiman')
    parser.add_argument('--caiman_intial_temp', type=bool, default=False, help='Use modified method to compute initial template using caiman')
    parser.add_argument('--jorncorre_intial_temp', type=bool, default=False, help='Use modified method to compute initial template using jnormcorre')

    # Parse the arguments
    args = parser.parse_args()

    run(args.input, args.output, args.suite2p, args.caiman, args.writetiff, args.caiman_intial_temp, args.jorncorre_intial_temp)