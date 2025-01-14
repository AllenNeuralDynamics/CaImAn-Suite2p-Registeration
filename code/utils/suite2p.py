import shutil
import os
import suite2p
import numpy as np
import h5py
from pathlib import Path
from tifffile import TiffWriter, imwrite

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

def suite2pRegistration(fn, output_path_temp, folder_number, output_path_suite2p):
    print('fn suite2p', fn)
    
    # Load Suite2p options
    ops = np.load("../code/utils/ops.npy", allow_pickle=True).item()
    print('ops:', ops)
    # Configure Suite2p database
    db = {
        'look_one_level_down': False,  # Do not search subfolders
        'data_path': [Path(fn).parent],  # Folder containing input TIFFs
        'tiff_list': [os.path.basename(fn)],  # List of TIFF files to process
        'save_path0': os.path.join(output_path_temp, folder_number),  # Temporary output path for Suite2p
        'maxregshift': 0.5, # MaxShift = min(Lx,Ly) * maxregshift; Caiman and Strip has a paramater called Maxshift but suite2p doesnt.
        #'nonrigid' : True, # TODO: Throws errors in some movies. 
    }

    # Run Suite2p registration
    opsEnd = suite2p.run_s2p(ops=ops, db=db)

    # Load registered binary data (data.bin)
    reg_file = opsEnd['reg_file']  # Path to registered binary file
    Ly, Lx = opsEnd['Ly'], opsEnd['Lx']  # Image dimensions (height and width)
    f_reg = suite2p.io.BinaryFile(Ly=Ly, Lx=Lx, filename=reg_file).data

    # Save registered data as a multi-page TIFF file
    tif_path = output_path_suite2p + '.tif'
    imwrite(tif_path, f_reg)  
    #with TiffWriter(tif_path, bigtiff=True) as tif:
    #    for frame in f_reg:
    #        frame[frame < 0] = 0  # Remove negative values (if any)
    #        tif.write(np.float32(frame))  # Convert to float32 and write to TIFF

    print(f"Registered movie saved as TIFF at {output_path_suite2p}")

    # Save motion correction offsets (optional)
    hdf5_path = output_path_suite2p + '.h5'
    with h5py.File(hdf5_path, 'w') as hdf:
        hdf.create_dataset('R', data=opsEnd['xoff'])  # Row offsets (x-direction)
        hdf.create_dataset('C', data=opsEnd['yoff'])  # Column offsets (y-direction)
        #hdf.create_dataset('R_1', data=opsEnd['xoff1'])  # Sub pixel shifts (x-direction); Needs #'nonrigid' : True
        #hdf.create_dataset('C_1', data=opsEnd['yoff1'])  # Sub pixel shifts (y-direction); #'nonrigid' : True

    print(f"Motion correction offsets saved in HDF5 format at {hdf5_path}")

    # Clean up temporary Suite2p files (optional)
    delete_suite2p_folder(os.path.dirname(reg_file))
