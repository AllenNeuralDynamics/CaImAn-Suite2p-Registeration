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
- No initial template (vanilla registration) where we allow CaImAn to generate the initial template and then let CaImAn motion correction over the whole movie
- StripRegisteration initial logic:
  Used the most correlated frames from the initial 1000 frames and averaged them. 
  Then used:
  - CaImAn template generation where we generate an initial template movie using CaImAn's motion correction function. 
  - JNormCorre template generation where we generate an initial template movie using JNormCorre's motion correction function. 


> [!WARNING]  
> CaImAn vanilla registration performs very poorly. 
