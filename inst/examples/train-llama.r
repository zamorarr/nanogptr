# following: https://blogs.rstudio.com/ai/posts/2023-05-25-llama-tensorflow-keras/
# pip install -U tensorflow_text==2.11.0

library(keras)
library(tensorflow)
library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

#model_path <- normalizePath("~/data/llama/7B")

# convert weights to numpy
# reticulate::py_install("torch", pip = TRUE)
#torch <- reticulate::import("torch", convert = FALSE)

# out of memory?
#write_weights <- function(path) {
#  pretrained_weights <- torch$load(
#    file.path(path, "consolidated.00.pth"),
#    map_location = "cpu"
#    )
#}

# tokenizer
tokenizer <- llama_tokenizer("~/data/llama/tokenizer.model")
tokenizer$tokenize("The best way to attract bees")
tokenizer$vocab_size()$numpy()

# params
#params <- jsonlite::read_json(file.path(model_path, "params.json"))

# lite params for testing
params <- list(
  dim = 20L,
  multiple_of = 5L,
  n_heads = 2L,
  n_layers = 7L,
  norm_eps = 1E-6,
  vocab_size = tokenizer$vocab_size()
)


# create model
model <- llama_model(params)
model$count_params()
plot(model, show_shapes = TRUE)

# compile
compile(
  model,
  #loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  jit_compile = TRUE)

# example
(c("The best way to attract bees", "The tree is green") |>
    tokenizer$tokenize())$to_tensor() |>
  token_embeddings() |>
  layer_multi_self_attention(num_heads = 32L, head_size = 128L) |>
  tf$shape()
