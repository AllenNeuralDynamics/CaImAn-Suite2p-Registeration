from tifffile import imread, imwrite
import os
from caiman.motion_correction import MotionCorrect
import h5py
import numpy as np
from tifffile import imread, imwrite
import os
from caiman.motion_correction import MotionCorrect
import caiman as cm
import h5py
from jnormcorre.motion_correction import MotionCorrect as jnormcorreMotionCorrect
import cv2
from scipy.cluster.hierarchy import linkage
import tifffile
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster

def read_tiff_file(fn):
    print("Reading:", fn)
    with tifffile.TiffFile(fn) as tif:
        imageData = tif.asarray()
        Ad = np.array(imageData, dtype=np.float32)

        if len(Ad.shape) == 3:
            Ad = np.reshape(
                Ad, (Ad.shape[0], 1, Ad.shape[1], Ad.shape[2])
            )  # Add channel info

        numChannels = Ad.shape[1]
    return Ad, numChannels

def use_stripRegisteration_first_template_generation(Yhp, dview, maxshift, path_template_list, caiman_template=True):
    # Reshape Yhp to 2D where each column is a frame
    reshaped_Yhp = Yhp.reshape(-1, Yhp.shape[2])

    # Calculate the correlation matrix
    rho = np.corrcoef(reshaped_Yhp.T)

    # Compute the distance matrix
    dist_matrix = 1 - rho

    # Ensure the distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Ensure the diagonal is zero (optional but often necessary for distance matrices)
    np.fill_diagonal(dist_matrix, 0)

    # Check if the matrix is symmetric
    is_symmetric = np.array_equal(dist_matrix, dist_matrix.T)
    print(f"Is the distance matrix symmetric? {is_symmetric}")

    # Perform hierarchical clustering using average linkage
    Z = linkage(squareform(dist_matrix), method="average")

    Z[:, :2] = np.ceil(Z[:, :2]).astype(int)

    # Z = Z[:, :-1] # Matlab produces just 3 columns

    # Define the cutoff value
    cutoff = 0.01

    # Define the minimum cluster size
    min_cluster_size = 100

    # Initialize an empty list for clusters
    clusters = []

    # Define the maximum cutoff value
    max_cutoff = 2.0

    # Initialize variables
    cutoff = 0.01
    min_cluster_size = 100
    clusters = []
    max_cutoff = 2.0

    while not clusters or all(len(cluster) < min_cluster_size for cluster in clusters):
        cutoff += 0.01
        if cutoff > max_cutoff:
            raise ValueError(
                f"Could not find a cluster with at least {min_cluster_size} samples"
            )

        # Perform clustering with the current cutoff
        T = fcluster(Z, cutoff, criterion="distance")

        # Group indices by cluster label
        clusters = [np.where(T == label)[0] for label in np.unique(T)]

    # Initialize max_mean_corr to negative infinity
    max_mean_corr = -np.inf

    # Iterate over each cluster
    for cluster_indices in clusters:
        # Check if the cluster size meets the minimum requirement
        if len(cluster_indices) >= min_cluster_size:
            # Compute the mean correlation within the cluster
            mean_corr = np.mean(np.mean(rho[np.ix_(cluster_indices, cluster_indices)]))

            # Update max_mean_corr and best_cluster if necessary
            if mean_corr > max_mean_corr:
                max_mean_corr = mean_corr
                best_cluster = cluster_indices

    if caiman_template:
        print("Using Caiman for template initial generate...")

        try:
            cv2.setNumThreads(0)
        except:
            pass

        max_shifts = (
            22,
            22,
        )  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
        strides = (48, 48)  # create a new patch every x pixels for pw-rigid correction
        overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
        max_deviation_rigid = (
            3  # maximum deviation allowed for patch with respect to rigid shifts
        )
        pw_rigid = (
            False  # flag for performing rigid or piecewise rigid motion correction
        )
        shifts_opencv = False  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
        border_nan = (
            True # replicate values along the boundary (if True, fill in with NaN)
        )

        # # Start the cluster (if a cluster already exists terminate it)
        # if 'dview' in locals():
        #     cm.stop_server(dview=dview)
        
        # c, dview, n_processes = cm.cluster.setup_cluster(
        #     backend='multiprocessing', n_processes=None, single_thread=False)

        print(dview)

        # Create a motion correction object
        mc = MotionCorrect(
            Yhp.transpose(2, 0, 1),
            dview=dview,
            max_shifts=max_shifts,
            shifts_opencv=shifts_opencv,
            nonneg_movie=True,
            border_nan=border_nan,
        )

        mc.motion_correct(
            template=np.mean(Yhp[:, :, best_cluster], axis=2), save_movie=True
        )
        m_rig = cm.load(mc.mmap_file)

        F = np.transpose(m_rig, (1, 2, 0))
        path_template_list.append(mc.mmap_file)

    else:
        print("Using JNormCorre for template initial generate...")

        corrector = jnormcorreMotionCorrect(
            lazy_dataset=Yhp.transpose(2, 0, 1),
            max_shifts=(maxshift, maxshift),  # Maximum allowed shifts in pixels
            strides=(48, 48),  # Patch dimensions for piecewise rigid correction
            overlaps=(24, 24),  # Overlap between patches
            max_deviation_rigid=3,  # Maximum deviation for rigid correction
            pw_rigid=False,
        )  # Number of frames to process in a batch

        frame_corrector, output_file = corrector.motion_correct(
            template=np.mean(Yhp[:, :, best_cluster], axis=2), save_movie=True
        )

        # Get the current working directory
        cwd = os.getcwd()

        # Get path to the template movie generated by jnormcorre
        path_template = os.path.join(cwd, " ".join(output_file))
        path_template_list.append(path_template)

        # Check if the path exists
        if os.path.exists(path_template):
            print(f"The template {path_template} exists.")
        else:
            print(f"The template {path_template} does not exist.")

        template,_ = read_tiff_file(
            path_template
        )  # TODO: Replace with tifffilereader

        F = template
        # F = np.transpose(template, (1, 2, 0))

    return F, path_template_list

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
def CaImAnRegistration(fname, output_path_caiman, use_caiman_template=False, use_jormcorre_template=False, output_shape=None, constant_values=0):
    
    print('output_path_caiman:', output_path_caiman)
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path_caiman)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        cv2.setNumThreads(0)
    except:
        pass

    # Debugging imread
    data = imread(fname)

    # Debugging channel processing
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
    border_nan = True    # replicate values along the boundary (if True, fill in with NaN)

    # Start the cluster (if a cluster already exists terminate it)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='multiprocessing', n_processes=None, single_thread=False)

    # Create a motion correction object using padded data
    mc = MotionCorrect(padded_data, dview=dview, max_shifts=max_shifts,
                       shifts_opencv=shifts_opencv, nonneg_movie=True,
                       border_nan=border_nan)

    if (not use_caiman_template) and (not use_jormcorre_template):
        print('Using caiman to generate intial template')
        # Perform motion correction
        mc.motion_correct(save_movie=True, template=None) # No template for vanilla caiman. 
    elif use_caiman_template:
        print('Using Hacked up version from stripReg to compute intial template. Using Caiman....')
        template, path_template_list = use_stripRegisteration_first_template_generation(np.transpose(padded_data[:,:, :1000], (1,2,0)), dview, maxshift = 50, path_template_list = [], caiman_template = True)
        mc.motion_correct(save_movie=True, template=np.nanmean(template, axis=2)) # Best cluster used to generate initial template movie using caiman.  
    elif use_jormcorre_template:
        print('Using Hacked up version from stripReg to compute intial template. Using Jnormcorre....')
        template, path_template_list = use_stripRegisteration_first_template_generation(np.transpose(padded_data[:,:, :1000], (1,2,0)), dview, maxshift = 50, path_template_list = [], caiman_template = False)
        mc.motion_correct(save_movie=True, template=np.nanmean(template[:,0,:,:], axis=0)) # Best cluster used to generate initial template movie using jnormcorre. 
            
    # Get and center shifts around zero
    coordinates = mc.shifts_rig
    x_shifts = [coord[0] for coord in coordinates]
    y_shifts = [coord[1] for coord in coordinates]
    
    # Now proceed with writing the HDF5 file
    with h5py.File(output_path_caiman + '.h5', 'w') as hdf:
        hdf.create_dataset('R', data=x_shifts)
        hdf.create_dataset('C', data=y_shifts)

    print('Saving tif file at', output_path_caiman)

    # Load corrected movie and save it to disk
    m_rig = cm.load(mc.mmap_file)
    corrected_data = m_rig.astype(np.float32)
    imwrite(output_path_caiman + ".tif", corrected_data)    
