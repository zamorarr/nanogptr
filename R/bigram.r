library(keras)
library(tensorflow)
library(tfdatasets)
library(glue)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

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
print(glue::glue("length of dataset in characters: {nchar(text)}"))

# first 1000 characters
cat(substr(text, 1, 1000))

# unique characters in this text
chars <- sort(unique(strsplit(text, "")[[1]]))
vocab_size <- length(chars)
print(glue_collapse(chars))
print(vocab_size)

# mapping from characters to integers
stoi <- setNames(seq_along(chars) - 1L, chars)
#itos <- chars
#encode <- function(x) lapply(strsplit(x, ""), function(s) unname(stoi[s])) # note not zero-indexed
#decode <- function(x) lapply(x, function(i) paste0(chars[i], collapse = ""))
encode <- function(x) unname(stoi[strsplit(x, "")[[1]]]) # note not zero-indexed
decode <- function(x) paste0(chars[x + 1L], collapse = "")
print(encode("hii there"))
print(decode(encode("hii there")))

# encode the entire dataset and store it in a tensor
data <- tf$constant(encode(text), dtype = tf$dtypes$int64)

# split into train/val sets
n <- as.integer(0.9*length(data))
train_data <- data[1:n]
val_data <- data[(n+1):length(data)]
stopifnot(length(train_data) + length(val_data) == length(data))

# block/context size
block_size <- 8
train_data[1:(block_size+1)]

x <- train_data[1:block_size]
y <- train_data[2:(block_size + 1)]
for (k in seq_len(block_size)) {
  context <- x[1:k]
  target <- y[k]
  cat("input:\n")
  print(context)
  cat("target:\n")
  print(target)
  cat("=============\n")
}

# context
tf$random$set_seed(100L)
#keras$utils$set_random_seed(100L)
batch_size <- 4L # number of sequences to process
block_size <- 8L # max context length for predictions

get_batch <- function(split) {
  # generate a small batch of data of inputs x and targets y
  data <- if (split == "train") train_data else val_data

  # choose {batch_size} random starting points
  ix <- tf$random$uniform(shape(batch_size), minval = 1L, maxval = length(data) - block_size, dtype = tf$dtypes$int32)

  # get context and targets
  x <- tf$stack(lapply(ix, \(i) data[i:(i + block_size - 1)]))
  y <- tf$stack(lapply(ix, \(i) data[(i + 1):(i + block_size)]))

  list(x = x, y = y)
}



c(xb, yb) %<-% get_batch("train")
cat("inputs:\n")
print(xb$shape)
print(xb)
cat("targets:\n")
print(yb$shape)
print(yb)
cat("=============\n")

for (b in seq_len(batch_size)) {
  for (k in seq_len(block_size)) {
    context <- xb[b, 1:k]
    target <- yb[b, k]
    cat("input:\n")
    print(context)
    cat("target:\n")
    print(target)
    cat("=============\n")
  }
}

# layers
#inputs <- layer_input(shape = c(block_size))
inputs <- layer_input(shape = list(NULL))
outputs <- inputs |> layer_embedding(vocab_size, vocab_size)
model <- keras_model(inputs, outputs, name = "bigram_model")

compile(
  model,
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(1E-3))

batch_size <- 128L
dataset <- tensor_slices_dataset(train_data) |>
  dataset_batch(block_size + 1, drop_remainder = TRUE) |>
  #dataset_take(3) |>
  dataset_map(function(x) {
    input_text <- x[1:block_size]
    target_text <- x[2:(block_size + 1)]
    list(input_text, target_text)
  }) |>
  dataset_shuffle(1000) |>
  dataset_batch(batch_size, drop_remainder = TRUE) |>
  dataset_prefetch()

history <- fit(
  model,
  #function(x) get_batch("train"),
  dataset,
  epochs = 5,
  #steps_per_epoch = 100
)

generate <- function(model, inputs, max_new_tokens = 100) {
  # inputs is (B,T) array of indices in current context
  for (i in seq_len(max_new_tokens)) {
    # get the predictions
    logits <- model(inputs) # {B, T, C}

    # focus on last context token (bigram model)
    logits <- logits[,dim(logits)[2],] # {B, C}

    # apply softmax
    #probs <- k_softmax(logits, axis = -1)

    # sample from distribution
    idx_next <- tf$random$categorical(logits, 1L) # {B, 1}

    # append sampled index to running sequence
    inputs <- keras::k_concatenate(list(inputs, idx_next), axis = 2) # {B, T+1}
  }

  # return final
  inputs
}

dummy_input <- tf$constant(0L, shape = shape(1, 1), dtype = tf$dtypes$int64)
cat(decode(generate(model, dummy_input, 100)$numpy()[1,]))
