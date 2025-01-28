# CaImAn-Suite2p-Registeration

A modular pipeline for motion correction and image registration of iGluSnFR data using Suite2p and CaImAn.

Syntax:
```
./run.sh --input <data_directory> --output <output_directory> --pipeline <caiman|suite2p> [--initial_temp <caiman_initial_temp|jorncorre_initial_temp>]
```

## Suite2p
The setting best for our iGluSnFR data is in [ops.npy](code/utils/ops.npy) 


## CaImAn
Initial Template Options:
- CaImAn template generation where we generate an initial template movie using CaImAn's motion correction function and then use its average as an initial template for CaImAn motion correction over the whole movie. 
- JNormCorre template generation where we generate an initial template movie using JNormCorre's motion correction function and then use its average as an initial template for CaImAn motion correction over the whole movie. 
- No initial template (vanilla registration) where we allow CaImAn to generate the initial template and then let CaImAn motion correction over the whole movie

> [!WARNING]  
> CaImAn vanilla registration performs very poorly. 
