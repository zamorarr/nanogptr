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
  vocab_size = as.vector(tokenizer$vocab_size())
)

# llama 7B params
params <- list(
  dim = 4096L,
  multiple_of = 256L,
  n_heads = 32L,
  n_layers = 32L,
  norm_eps = 1E-6,
  vocab_size = as.vector(tokenizer$vocab_size())
)

# check model fits in memory
with(params, llama_size(vocab_size, n_layers, n_heads, dim, multiple_of))

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
generate <- tf_function(function(model, inputs, max_new_tokens = 100) {
  block_size <- 2048L
  #withr::local_options(tensorflow.extract.style = "python")

  # inputs is (B,T) array of indices in current context
  #batch_size <- nrow(inputs)
  #batch_size <- tf$shape(inputs)[1]

  # create progress bar
  #pb <- progress::progress_bar$new(
  #  total = max_new_tokens,
  #  format = "[:bar] :current/:total (:percent) eta: :eta")

  tfautograph::ag_while_opts(shape_invariants = list(
    inputs = tf$TensorShape(list(1L, NULL))
    #i = tf$TensorShape(list())
  ))

  for (i in tf$range(as.integer(max_new_tokens))) {
    #for (i in seq_len(max_new_tokens)) {
    #pb$tick()

    # crop inputs to last block_size tokens
    context_size <- tf$shape(inputs)[2]
    start <- tf$maximum(context_size - block_size, 0L)
    inputs_cropped <- inputs[,start:context_size]

    # get the predictions
    logits <- model(inputs_cropped) # {B, T, C}

    # focus on last context token (bigram model)
    logits <- logits[,-1L,]
    #logits <- logits[,dim(logits)[2],] # {B, C}
    #logits <- matrix(logits, nrow = batch_size)

    # sample from distribution
    idx_next <- tf$random$categorical(logits, 1L) # {B, 1}

    # append sampled index to running sequence
    inputs <- keras::k_concatenate(list(inputs, idx_next), axis = 2) # {B, T+1}
  }

  # return final
  inputs
})

input <- tokenizer$tokenize("The best way to attract bees") |>
  tf$cast(dtype = tf$dtypes$int64) |>
  tf$expand_dims(0L)

generate(model, input, max_new_tokens = 10L) |>
  tf$cast(dtype = tf$dtypes$int32) |>
  tokenizer$detokenize() |>
  as.character()
