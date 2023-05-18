library(keras)
library(tensorflow)
library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

# hyper params
batch_size <- 12 # number of sequences to process
block_size <- 64L # max context length for predictions
embed_size <- 128L
attention_heads <- 4L
#attention_size <- 6L # size of attention output features = embed_size %/% attention_heads
num_transformers <- 4L
dropout_rate = 0.0
learning_rate <- 3E-4
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

dataset_train <- build_pipeline(train_data)
dataset_val <- build_pipeline(val_data)

# custom layers
layer_pos_embedding <- new_layer_class(
  "PosEmbedding",
  initialize = function(input_dim, output_dim, ...) {
    super$initialize(...)
    self$input_dim <- input_dim
    self$output_dim <- output_dim
    self$embedding <- layer_embedding(input_dim = input_dim, output_dim = output_dim)
  },

  call = function(inputs) {
    # get context size of inputs (can be less than layer context size)
    shp <- tf$shape(inputs)
    input_dim <- shp[2]

    # create positions input
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

  get_config = function() {
    config <- super$get_config()
    config$input_dim <- self$input_dim
    config$output_dim <- self$output_dim
    config
  }
)

layer_self_attention <- new_layer_class(
  "SelfAttention",
  initialize = function(max_context_size, output_size, dropout_rate = 0, ...) {
    super$initialize(...)
    self$key <- layer_dense(units = output_size, use_bias = FALSE)
    self$query <- layer_dense(units = output_size, use_bias = FALSE)
    self$value <- layer_dense(units = output_size, use_bias = FALSE)

    self$tril_mask <- self$create_tril(c(max_context_size, max_context_size))
    self$w_norm <- tf$sqrt(tf$cast(output_size, tf$dtypes$float32))

    self$dropout <- layer_dropout(rate = dropout_rate)

    self$max_context_size <- max_context_size
    self$output_size <- as.integer(output_size)
    self$dropout_rate <- dropout_rate
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

    # dropout
    w <- self$dropout(w)

    # calculate weighted values
    tf$matmul(w, v) # {B,L,L} x {B,L,A} = {B,L,A}
  },

  get_config = function() {
    config <- super$get_config()
    config$max_context_size <- self$max_context_size
    config$output_size <- self$output_size
    config$dropout_rate <- self$dropout_rate
    config
  }
)

layer_multi_self_attention <- new_layer_class(
  "MultiSelfAttention",
  initialize = function(num_heads, max_context_size, attention_size, dense_units,
                        dropout_rate = 0, ...) {
    super$initialize(...)

    # layers
    self$heads <- lapply(seq_len(num_heads), function(i) {
      layer_self_attention(max_context_size = max_context_size, output_size = attention_size, dropout_rate = dropout_rate)
    })
    self$proj <- layer_dense(units = dense_units)
    self$dropout <- layer_dropout(rate = dropout_rate)

    # config
    self$num_heads <- num_heads
    self$max_context_size <- max_context_size
    self$attention_size <- attention_size
    self$dense_units <- dense_units
    self$dropout_rate <- dropout_rate
  },


  call = function(input) {
    out <- lapply(self$heads, function(h) h(input))
    out <- tf$concat(out, axis = -1L)
    out |> self$proj() |> self$dropout()
  },

  get_config = function() {
    config <- super$get_config()
    config$num_heads <- self$num_heads
    config$max_context_size <- self$max_context_size
    config$attention_size <- self$attention_size
    config$dense_units <- self$dense_units
    config$dropout_rate <- self$dropout_rate
    config
  }
)

layer_feedforward <- new_layer_class(
  "FeedForward",
  initialize = function(units, dropout_rate = 0, ...) {
    super$initialize(...)

    # layers
    # factor of 4 from Attention is All you Need paper
    self$dense <- layer_dense(units = 4*units, activation = "relu")
    self$proj <- layer_dense(units = units)
    self$dropout <- layer_dropout(rate = dropout_rate)

    # config
    self$units <- units
    self$dropout_rate <- dropout_rate
  },

  call = function(input) {
    input |>
      self$dense() |>
      self$proj() |>
      self$dropout()
  },

  get_config = function() {
    config <- super$get_config()
    config$units <- self$units
    config$dropout_rate <- self$dropout_rate
    config
  }
)

layer_transformer_block <- new_layer_class(
  "TransformerBlock",
  initialize = function(num_heads, max_context_size, embed_size, dropout_rate = 0, ...) {
    super$initialize(...)

    # self attention
    attention_size <- embed_size %/% num_heads
    self$sa <- layer_multi_self_attention(
      num_heads = num_heads,
      max_context_size = max_context_size,
      attention_size = attention_size,
      dense_units = embed_size,
      dropout_rate = dropout_rate)

    # feed forward
    self$ffwd <- layer_feedforward(units = embed_size, dropout_rate = dropout_rate)

    # layer norms
    self$ln1 <- layer_layer_normalization(axis = -1L)
    self$ln2 <- layer_layer_normalization(axis = -1L)

    # config
    self$num_heads <- num_heads
    self$max_context_size <- max_context_size
    self$embed_size <- embed_size
    self$dropout_rate <- dropout_rate
  },

  call = function(input) {
    # residual connection
    x <- input + self$sa(self$ln1(input))
    x <- x + self$ffwd(self$ln2(x))
    x
  },

  get_config = function() {
    config <- super$get_config()
    config$num_heads <- self$num_heads
    config$max_context_size <- self$max_context_size
    config$embed_size <- self$embed_size
    config$dropout_rate <- self$dropout_rate
    config
  }
)

# model
inputs <- layer_input(shape = list(NULL), name = "input")
token_embeds <- inputs |> layer_embedding(vocab_size, embed_size, name = "token_embeddings")
pos_embeds <- inputs |> layer_pos_embedding(block_size, embed_size, name = "position_embeddings")
input_embeds <- layer_add(list(token_embeds, pos_embeds))

#sa_head <- input_embeds |> layer_self_attention(block_size, attention_size, name = "self_attention")
#sa_heads <- input_embeds |> layer_multi_self_attention(
#  num_heads = attention_heads,
#  block_size = block_size,
#  attention_size = attention_size %/% attention_heads,
#  units = embed_size,
#  name = "self_attention")

transformer_blocks <- input_embeds
for (i in seq_len(num_transformers)) {
  transformer_blocks <- transformer_blocks |>
    layer_transformer_block(
      attention_heads, block_size, embed_size, dropout_rate = dropout_rate,
      name = paste0("transformer_", i))
}

output <- transformer_blocks |>
  layer_layer_normalization(axis = -1L) |>
  layer_dense(units = vocab_size, name = "linear_head")

model <- keras_model(inputs, output, name = "gpt_model")


#model((dataset |> dataset_take(1) |> dataset_collect())[[1]][[1]]) |> dim()
#model((dataset |> dataset_take(1) |> dataset_collect())[[1]][[1]][,1:3]) |> dim()
plot(model, show_shapes = TRUE)

compile(
  model,
  loss = loss_sparse_categorical_crossentropy(from_logits = TRUE),
  optimizer = optimizer_adam(learning_rate))

# callbacks
lr_reduce <- callback_reduce_lr_on_plateau(factor = 0.5, patience = 3,
                                           min_lr = learning_rate/10)
early_stopping <- callback_early_stopping(patience = 5)

# fit
history <- fit(model, dataset_train, epochs = max_epochs, validation_data = dataset_val,
               callbacks = list(lr_reduce, early_stopping))

generate <- function(model, inputs, max_new_tokens = 100) {
  # inputs is (B,T) array of indices in current context
  batch_size <- nrow(inputs)

  # create progress bar
  pb <- progress::progress_bar$new(
    total = max_new_tokens,
    format = "[:bar] :current/:total (:percent) eta: :eta")

  for (i in seq_len(max_new_tokens)) {
    pb$tick()
    # crop inputs to last block_size tokens
    context_size <- ncol(inputs)
    start <- max(context_size - block_size + 1,1)
    inputs_cropped <- inputs[,start:context_size]

    # get the predictions
    #logits <- model(inputs_cropped) # {B, T, C}
    logits <- predict(model, inputs_cropped)

    # focus on last context token (bigram model)
    logits <- logits[,dim(logits)[2],] # {B, C}
    logits <- matrix(logits, nrow = batch_size)

    # sample from distribution
    idx_next <- tf$random$categorical(logits, 1L) # {B, 1}

    # append sampled index to running sequence
    inputs <- k_concatenate(list(inputs, idx_next), axis = 2) # {B, T+1}
  }

  # return final
  inputs
}

# dummy input
dummy_prompt <- encode("BOBBY:\n")
dummy_input <- tf$constant(dummy_prompt, shape = shape(1, length(dummy_prompt)), dtype = tf$dtypes$int64)
dummy_output <- generate(model, dummy_input, 100)
cat(decode(dummy_output$numpy()[1,]))

# write larger output
large_output <- generate(model, dummy_input, 1000)
writeLines(decode(large_output$numpy()[1,]), "data/more.txt")

# save model
#save_model_weights_tf(model, "checkpoints/gpt.ckpt")
#save_model_tf(model, "checkpoints/small")

# load model
#model <- load_model_tf("checkpoints/small")
#evaluate(model, dataset_val)
