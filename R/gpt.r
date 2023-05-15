library(keras)
library(tensorflow)
library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

# hyper params
batch_size <- 128L # number of sequences to process
block_size <- 8L # max context length for predictions
attention_heads <- 4L
attention_size <- 6L # size of attention output features
learning_rate <- 1E-3
embed_size <- 32L
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

# build data pipeline
build_pipeline <- function(x) {
  tensor_slices_dataset(x) |>
    dataset_batch(block_size + 1, drop_remainder = TRUE) |>
    dataset_map(function(x) {
      input_text <- x[1:block_size]
      target_text <- x[2:(block_size + 1)]
      list(input_text, target_text)
    }) |>
    dataset_shuffle(1000) |>
    dataset_batch(batch_size, drop_remainder = TRUE) |>
    dataset_prefetch()
}

dataset <- build_pipeline(train_data)
dataset_val <- build_pipeline(val_data)

# custom layers
layer_pos_embedding <- new_layer_class(
  "PosEmbedding",
  initialize = function(input_dim, output_dim, ...) {
    super$initialize(...)
    #self$input_dim <- as.integer(input_dim)
    #self$output_dim <- as.integer(output_dim)
    self$input_dim <- input_dim
    self$output_dim <- output_dim
    #self$embedding <- keras$layers$Embedding(input_dim, output_dim)
    self$embedding <- layer_embedding(input_dim = input_dim, output_dim = output_dim)
  },

  call = function(inputs) {
    shp <- tf$shape(inputs)
    input_dim <- shp[2]

    positions <- tf$range(input_dim)
    embeddings <- self$embedding(positions)

    # trim to match the length of input, which
    # might be less than length of input_dim of layer
    embeddings <- tf$slice(embeddings, c(0L, 0L), c(input_dim, self$output_dim))
    #embeddings <- embeddings[1:input_dim,]

    # broadcast to add missing batch dimensions
    new_shp <- tf$concat(list(shp, list(self$output_dim)), axis = -1L)
    tf$broadcast_to(embeddings, new_shp)
  },

  get_config = NULL
)

layer_self_attention <- new_layer_class(
  "SelfAttention",
  initialize = function(block_size, output_size, ...) {
    super$initialize(...)
    self$key <- layer_dense(units = output_size, use_bias = FALSE)
    self$query <- layer_dense(units = output_size, use_bias = FALSE)
    self$value <- layer_dense(units = output_size, use_bias = FALSE)

    self$tril_mask <- self$create_tril(c(block_size, block_size))
    self$w_norm <- tf$sqrt(tf$cast(output_size, tf$dtypes$float32))

    self$output_size <- as.integer(output_size)
  },

  create_tril = function(shape) {
    # creates lower triangular boolean mask over last 2 dimensions
    row_index <- tf$cumsum(tf$ones(shape = shape, dtype = tf$int32), axis = -2L)
    col_index <- tf$cumsum(tf$ones(shape = shape, dtype = tf$int32), axis = -1L)
    tf$greater_equal(row_index, col_index)
    #row_index <- k_cumsum(k_ones(shape, dtype = tf$dtypes$int32), axis = -2)
    #col_index <- k_cumsum(k_ones(shape, dtype = tf$dtypes$int32), axis = -1)
    #k_greater_equal(row_index, col_index)
  },

  call = function(inputs) {
    # inputs {B, L, V}
    shp <- tf$shape(inputs)
    context_size <- shp[2]

    # project inputs
    k <- self$key(inputs) # {B, L, A} A = attention output_size
    q <- self$query(inputs) # {B, L, A}
    v <- self$value(inputs) # {B, L, A}

    # calculate affinities
    kt <- k_permute_dimensions(k, c(1, 3, 2)) # {B, A, L}
    w <- tf$matmul(q, kt)/self$w_norm # {B,L,A} x {B,A,L} = {B, L, L}

    # mask for casual self-attention (cannot see tokens in front of it)
    # trim tril to {context_size,context_size}. Can be up to {block_size, block_size}
    tril_mask <- tf$slice(self$tril_mask, c(0L, 0L), c(context_size, context_size))
    w <- tf$where(!tril_mask, tf$fill(tf$shape(w), -Inf), w)
    w <- k_softmax(w)

    # calculate weighted values
    tf$matmul(w, v) # {B,L,L} x {B,L,A} = {B,L,A}
  }
)

layer_multi_self_attention <- new_layer_class(
  "MultiSelfAttention",
  initialize = function(num_heads, block_size, attention_size, units, ...) {
    super$initialize(...)
    self$heads <- lapply(seq_len(num_heads), function(i) {
      layer_self_attention(block_size = block_size, output_size = attention_size)
    })
    self$proj <- layer_dense(units = units)
  },


  call = function(input) {
    out <- lapply(self$heads, function(h) h(input))
    out <- tf$concat(out, axis = -1L)
    self$proj(out)
  }
)

layer_feedforward <- new_layer_class(
  "FeedForward",
  initialize = function(units, ...) {
    super$initialize(...)
    self$dense <- layer_dense(units = 4*units, activation = "relu")
    self$proj <- layer_dense(units = units)
  },
  call = function(input) {
    input |>
      self$dense() |>
      self$proj()
  }
)

layer_transformer_block <- new_layer_class(
  "TransformerBlock",
  initialize = function(num_heads, embed_size, context_size, attention_size, ...) {
    super$initialize(...)
    attention_size <- embed_size %/% num_heads
    self$sa <- layer_multi_self_attention(
      num_heads = num_heads,
      block_size = context_size,
      attention_size = attention_size,
      units = embed_size)
    self$ffwd <- layer_feedforward(units = embed_size)
  },
  call = function(input) {
    # residual connection
    x <- input + self$sa(input)
    x <- x + self$ffwd(x)
    x
  }
)

# model
#inputs <- layer_input(shape = c(block_size), name = "input")
attention_size <- embed_size

inputs <- layer_input(shape = list(NULL), name = "input")
token_embeds <- inputs |> layer_embedding(vocab_size, embed_size, name = "token_embeddings")
pos_embeds <- inputs |> layer_pos_embedding(block_size, embed_size, name = "position_embeddings")
token_pos_embeds <- layer_add(list(token_embeds, pos_embeds))
#sa_head <- token_pos_embeds |> layer_self_attention(block_size, attention_size, name = "self_attention")
#sa_heads <- token_pos_embeds |> layer_multi_self_attention(
#  num_heads = attention_heads,
#  block_size = block_size,
#  attention_size = attention_size %/% attention_heads,
#  units = embed_size,
#  name = "self_attention")

sa_blocks <- token_pos_embeds |>
  layer_transformer_block(4, embed_size, block_size) |>
  layer_transformer_block(4, embed_size, block_size) |>
  layer_transformer_block(4, embed_size, block_size)

lm_head <- sa_blocks |> layer_dense(units = vocab_size, name = "linear_head")
model <- keras_model(inputs, lm_head, name = "gpt_model")


#model((dataset |> dataset_take(1) |> dataset_collect())[[1]][[1]]) |> dim()
#model((dataset |> dataset_take(1) |> dataset_collect())[[1]][[1]][,1:3]) |> dim()
plot(model, show_shapes = TRUE)

compile(
  model,
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(learning_rate))

history <- fit(model, dataset, epochs = max_epochs, validation_data = dataset_val)

generate <- function(model, inputs, max_new_tokens = 100) {
  # inputs is (B,T) array of indices in current context
  for (i in seq_len(max_new_tokens)) {
    # crop inputs to last block_size tokens
    context_size <- ncol(inputs)
    start <- max(context_size - block_size + 1,1)
    inputs_cropped <- inputs[,start:context_size]

    # get the predictions
    logits <- model(inputs_cropped) # {B, T, C}

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
