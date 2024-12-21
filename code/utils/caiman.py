import numpy as np
from tifffile import imread
import caiman as cm
import os
from caiman.motion_correction import MotionCorrect
import h5py

# Padding functions
def pad_array_center(input_array, output_shape, constant_values):
    input_shape = input_array.shape
    pad_width = [(0, 0)] * len(input_shape)
    
    for i in range(len(input_shape)):
        total_pad = output_shape[i] - input_shape[i]
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width[i] = (pad_before, pad_after)
    
    padded_array = np.pad(input_array, pad_width, mode='constant', constant_values=constant_values)
    return padded_array

def pad_movie(movie, output_shape, constant_values):
    return np.array(list(map(lambda arr: pad_array_center(arr, output_shape, constant_values), movie)))

import numpy as np

def process_image_channel(data):
    """
    Processes multi-channel or single-channel image data.
    
    Args:
        data (numpy.ndarray): Input image data
    
    Returns:
        numpy.ndarray: Processed image data with first channel extracted and squeezed
    """
    # Check input dimensionality
    if data.ndim == 4 and data.shape[1] > 1:
        # Multi-channel data: extract first channel
        data = data[:, 1, :, :]
    
    # Ensure final shape is (height, width, depth)
    if data.ndim == 4:
        data = data.squeeze()
    
    return data

# Main CaImAn registration function with padding
def CaImAnRegistration(fname, output_path_caiman, output_shape=None, constant_values=0):
    print('fname', fname)
    try:
        cv2.setNumThreads(0)
    except:
        pass

    # Load the movie data
    data = imread(fname)  # TODO: Add logic for h5 as well
    data = process_image_channel(data)
   
    # Determine output shape for padding if not provided
    if output_shape is None:
        output_shape = data.shape[1:]  # Use existing frame dimensions if no output shape is specified

    # Pad the movie frames to the desired shape
    padded_data = pad_movie(data, output_shape, constant_values)

    max_shifts = (22, 22)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
    strides = (48, 48)     # create a new patch every x pixels for pw-rigid correction
    overlaps = (24, 24)    # overlap between patches (size of patch strides+overlaps)
    max_deviation_rigid = 3   # maximum deviation allowed for patch with respect to rigid shifts
    pw_rigid = False       # flag for performing rigid or piecewise rigid motion correction
    shifts_opencv = False  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
    border_nan = 'copy'    # replicate values along the boundary (if True, fill in with NaN)

    # Start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # Create a motion correction object using padded data
    mc = MotionCorrect(padded_data, dview=dview, max_shifts=max_shifts,
                       shifts_opencv=shifts_opencv, nonneg_movie=True,
                       border_nan=border_nan)

    # Perform motion correction
    mc.motion_correct(save_movie=True)
    
    # Load corrected movie and save it to disk
    # m_rig = cm.load(mc.mmap_file)
    # m_rig.save(output_path_caiman)

    # Get and center shifts around zero
    coordinates = mc.shifts_rig
    x_shifts = [coord[0] for coord in coordinates]
    y_shifts = [coord[1] for coord in coordinates]
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path_caiman)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Now proceed with writing the HDF5 file
    with h5py.File(output_path_caiman + '.h5', 'w') as hdf:
        hdf.create_dataset('R', data=x_shifts)
        hdf.create_dataset('C', data=y_shifts)

