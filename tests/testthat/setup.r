library(keras)
library(tensorflow)
#library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")

#option(tensorflow.extract.warn_tensors_passed_asis = FALSE)
