import shutil
import os
import suite2p
import numpy as np
import h5py
from pathlib import Path
from tifffile import tifffile

def remove_readonly(func, path, excinfo):
    """Clear the readonly bit and reattempt the removal"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

# Ensure complete deletion of suite2p folder
def delete_suite2p_folder(suite2p_path):
    try:
        # Force delete the entire suite2p directory
        shutil.rmtree(suite2p_path, onerror=remove_readonly)
        print(f"Successfully deleted {suite2p_path}")
    except Exception as e:
        print(f"Error deleting suite2p folder: {e}")

def suite2pRegistration(fn, output_path_temp, folder_number,output_path_suite2p):

  print('fn suite2p', fn)
  ops = np.load("../code/utils/ops.npy", allow_pickle = True).item()
  
  db = {
    # 'h5py': [], # a single h5 file path
    # 'h5py_key': 'data',
    'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
    'data_path': [Path(fn).parent], 
                  # a list of folders with tiffs 
                  # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)        
    # 'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
  #   'fast_disk': 'C:/BIN', # string which specifies where the binary file will be stored (should be an SSD)
    'tiff_list': [os.path.basename(fn)], # list of tiffs in folder * data_path *!
    'save_path0': os.path.join(output_path_temp, folder_number) # TODO: make sure output_path is only str
  }

  opsEnd = suite2p.run_s2p(ops=ops, db=db)

  # Read in raw tif corresponding 
  # f_raw = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=fname)
  # Create a binary file we will write our registered image to 
  # f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename= opsEnd['save_path'] + '/data.bin', n_frames = n_time).data # Set registered binary file to have same n_frames

  # Delete the temporary file created by suite2p otherwise registration will not take place for the next trial
  delete_suite2p_folder(os.path.dirname(opsEnd["reg_file"]))

  # with tifffile.TiffWriter(output_path_suite2, bigtiff=True) as tif:
  # f_reg[f_reg < 0] = 0
  # f_reg = np.uint16(f_reg)
  # tifffile.imwrite(output_path_suite2p, f_reg)
  # f_reg.write_tiff(output_path_suite2p)

  # Create an HDF5 file and save the NumPy array
  with h5py.File(output_path_suite2p+'.h5', 'w') as hdf:
      hdf.create_dataset('R', data=opsEnd['xoff'])
      hdf.create_dataset('C', data=opsEnd['yoff'])

  # Center the shifts to zero 
  # return opsEnd['xoff'], opsEnd['yoff']