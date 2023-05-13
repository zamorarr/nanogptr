library(keras)
library(tensorflow)
library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

# hyper params
batch_size <- 128L # number of sequences to process
block_size <- 8L # max context length for predictions
learning_rate <- 1E-3
max_epochs <- 10L

# get tiny shakespeare dataset
input_path <- "data/input.txt"
if (!file.exists(input_path)) {
  if (!dir.exists(dirname(input_path))) dir.create(dirname(input_path), recursive = TRUE)
  download.file(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    input_path
  )
}

# read it in
text <- readr::read_file(input_path)

# unique characters in this text
chars <- sort(unique(strsplit(text, "")[[1]]))
vocab_size <- length(chars)

# mapping from characters to integers
stoi <- setNames(seq_along(chars) - 1L, chars)
encode <- function(x) unname(stoi[strsplit(x, "")[[1]]]) # note not zero-indexed
decode <- function(x) paste0(chars[x + 1L], collapse = "")

# encode the entire dataset and store it in a tensor
data <- tf$constant(encode(text), dtype = tf$dtypes$int64)

# split into train/val sets
n <- as.integer(0.9*length(data))
train_data <- data[1:n]
val_data <- data[(n+1):length(data)]
stopifnot(length(train_data) + length(val_data) == length(data))

# layers
#inputs <- layer_input(shape = c(block_size))
inputs <- layer_input(shape = list(NULL))
outputs <- inputs |> layer_embedding(vocab_size, vocab_size)
model <- keras_model(inputs, outputs, name = "bigram_model")

compile(
  model,
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(learning_rate))

dataset <- tensor_slices_dataset(train_data) |>
  dataset_batch(block_size + 1, drop_remainder = TRUE) |>
  dataset_map(function(x) {
    input_text <- x[1:block_size]
    target_text <- x[2:(block_size + 1)]
    list(input_text, target_text)
  }) |>
  dataset_shuffle(1000) |>
  dataset_batch(batch_size, drop_remainder = TRUE) |>
  dataset_prefetch()

history <- fit(model, dataset, epochs = max_epochs)

generate <- function(model, inputs, max_new_tokens = 100) {
  # inputs is (B,T) array of indices in current context
  for (i in seq_len(max_new_tokens)) {
    # get the predictions
    logits <- model(inputs) # {B, T, C}

    # focus on last context token (bigram model)
    logits <- logits[,dim(logits)[2],] # {B, C}

    # sample from distribution
    idx_next <- tf$random$categorical(logits, 1L) # {B, 1}

    # append sampled index to running sequence
    inputs <- k_concatenate(list(inputs, idx_next), axis = 2) # {B, T+1}
  }

  # return final
  inputs
}

# dummy input
dummy_input <- tf$constant(encode("A"), shape = shape(1, 1), dtype = tf$dtypes$int64)
cat(decode(generate(model, dummy_input, 100)$numpy()[1,]))
