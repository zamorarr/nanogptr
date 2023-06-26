# compare huggingface llama weights to original
# this will help know how to convert between formats

library(dplyr)
library(stringr)
library(glue)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")

torch <- reticulate::import("torch", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)

hf_path <- "~/data/llama-hf-np/7B"
ll_path <- "~/data/llama-np/7B"

list.files(ll_path)

# TO CHECK
weight_names <- c(
  "norm.weight.npy",
  "output.weight.npy",
  "tok_embeddings.weight.npy",
  "layers.0.attention_norm.weight.npy",
  #"layers.0.attention.inner_attention.rope.freqs.npy",
  "layers.0.attention.wk.weight.npy", # FALSE
  "layers.0.attention.wo.weight.npy",
  "layers.0.attention.wq.weight.npy",
  "layers.0.attention.wv.weight.npy",
  "layers.0.feed_forward.w1.weight.npy",
  "layers.0.feed_forward.w2.weight.npy",
  "layers.0.feed_forward.w3.weight.npy",
  "layers.0.ffn_norm.weight.npy"
)

#weight_names <- intersect(list.files(ll_path), list.files(hf_path))

are_equal <- vapply(weight_names, function(wn) {
  message(wn)
  w_hf <- np$load(normalizePath(file.path(hf_path, wn), mustWork = TRUE))
  w_ll <- np$load(normalizePath(file.path(ll_path, wn), mustWork = TRUE))
  reticulate::py_to_r(np$array_equal(w_hf, w_ll))
}, FUN.VALUE = logical(1))


# find out which are not equal
names(are_equal)[which(!are_equal)]

# as expected, wk and wq are not the same because of the permutation
# let's see if we can fix that
mdim <- 4096L
n_heads <- 32L
head_size <- mdim %/% n_heads

# go from llama weights to huggingface weights
permute <- function(w) {
  w |>
    np$reshape(as.integer(c(n_heads, head_size %/% 2, 2, mdim))) |>
    np$transpose(c(0L, 2L, 1L, 3L)) |>
    np$reshape(c(mdim, mdim))
}

# go from huggingface weights to llama weights
unpermute <- function(w) {
  w |>
    np$reshape(as.integer(c(n_heads, 2, head_size %/% 2, mdim))) |>
    np$transpose(c(0L, 2L, 1L, 3L)) |>
    np$reshape(c(mdim, mdim))
}

q_hf <- np$load(normalizePath(file.path(hf_path, "layers.0.attention.wq.weight.npy"), mustWork = TRUE))
q_ll <- np$load(normalizePath(file.path(ll_path, "layers.0.attention.wq.weight.npy"), mustWork = TRUE))
k_hf <- np$load(normalizePath(file.path(hf_path, "layers.0.attention.wk.weight.npy"), mustWork = TRUE))
k_ll <- np$load(normalizePath(file.path(ll_path, "layers.0.attention.wk.weight.npy"), mustWork = TRUE))

reticulate::py_to_r(np$array_equal(q_hf, q_ll))
reticulate::py_to_r(np$array_equal(q_hf, permute(q_ll)))
reticulate::py_to_r(np$array_equal(unpermute(q_hf), q_ll))
reticulate::py_to_r(np$array_equal(k_hf, k_ll))
reticulate::py_to_r(np$array_equal(k_hf, permute(k_ll)))
reticulate::py_to_r(np$array_equal(unpermute(k_hf), k_ll))

#np$matmul(q_hf, k_hf)
#np$matmul(q_ll, k_ll)
