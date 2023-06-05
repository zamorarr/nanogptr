# following: https://blogs.rstudio.com/ai/posts/2023-05-25-llama-tensorflow-keras/

library(keras)
library(tensorflow)
library(tfdatasets)

Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")
tf$test$is_gpu_available()

model_path <- normalizePath("~/data/llama/7B")

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

# params
#params <- jsonlite::read_json(file.path(model_path, "params.json"))

# lite params for testing
params <- list(
  dim = 20L,
  multiple_of = 5L,
  n_heads = 2L,
  n_layers = 7L,
  norm_eps = 1E-6,
  vocab_size = -1L
)

# tokenizer
# pip install -U tensorflow_text==2.11.0
tf_text <- reticulate::import("tensorflow_text")
tokenizer <- tf_text$SentencepieceTokenizer(
  tf$io$gfile$GFile(normalizePath("~/data/llama/tokenizer.model"), "rb")$read(),
  add_bos = TRUE,
  add_eos = FALSE
)

tokenizer$tokenize("The best way to attract bees")
tokenizer$vocab_size()$numpy()

# custom layers
layer_rmsnorm <- new_layer_class(
  "RMSNorm",
  initialize = function(eps = 1E-6, ..., block_id = NULL, feeds_into = NULL) {
    super$initialize(...)
    self$eps <- eps
    self$block_id <- block_id
    self$feeds_into <- feeds_into
  },

  build = function(input_shape) {
    # input_shape = (batch_size, seqlen, model_dim)
    # w_shape = (1, 1, model_dim)
    size <- length(input_shape)
    model_dim <- tf$constant(tail(input_shape, 1), shape = shape(1))
    paddings <- tf$constant(array(c(size - 1L, 0L), dim = c(1, 2)))
    w_shape = tf$pad(model_dim, paddings, constant_values = 1L)
    self$w <- self$add_weight(
      shape = w_shape,
      initializer = "ones",
      trainable = TRUE)
  },

  rrms = function(x) {
    # reciprocal root mean square along last axis
    # same as dividing by std dev? assuming zero mean
    # x = {batch_size, seqlen, model_dim}
    x |>
      tf$math$square() |> # {B, T, D}
      tf$reduce_mean(axis = -1L, keepdims = TRUE) |> # {B, T, 1}
      tf$math$add(self$eps) |> # {B, T, 1}
      tf$math$rsqrt() # {B, T, 1}
  },

  call = function(x) {
    x * self$rrms(x) * self$w
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("eps", "block_id", "feeds_into")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

layer_feedforward <- new_layer_class(
  "FeedForward",
  initialize = function(hidden_dim, multiple_of = 256L, ..., block_id = NULL) {
    super$initialize(...)

    # round hidden_dim
    hidden_dim <- as.integer(hidden_dim * (2/3))
    hidden_dim <- (hidden_dim + multiple_of - 1) %/% multiple_of
    hidden_dim <- hidden_dim * multiple_of

    # config
    self$hidden_dim <- hidden_dim
    self$block_id <- block_id
  },

  build = function(input_shape) {
    output_dim <- tail(input_shape, 1)
    # layers
    self$w1 <- layer_dense(units = self$hidden_dim, use_bias = FALSE)
    self$w2 <- layer_dense(units = output_dim, use_bias = FALSE)
    self$w3 <- layer_dense(units = self$hidden_dim, use_bias = FALSE)
    super$build(input_shape)
  },

  call = function(x) {
    # SwiGLU
    self$w2(tf$nn$silu(self$w1(x)) * self$w3(x))
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("hidden_dim", "block_id")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

make_mask <- function(seqlen, dtype = k_floatx()) {
  x <- tf$range(seqlen)
  row_index <- x[, tf$newaxis]
  col_index <- x[tf$newaxis, ]
  mask <- tf$where(
    row_index < col_index,
    tf$constant(-Inf, dtype = dtype),
    tf$constant(0, dtype = dtype)
  )

  # {1, 1, seqlen, seqlen}
  mask[tf$newaxis, tf$newaxis, , ]
}

layer_multi_self_attention <- new_layer_class(
  "MultiSelfAttention",
  initialize = function(num_heads, head_size, ..., block_id = NULL) {
    super$initialize(...)

    # layers
    units <- num_heads * head_size
    self$wq <- layer_dense(units = units, use_bias = FALSE)
    self$wk <- layer_dense(units = units, use_bias = FALSE)
    self$wv <- layer_dense(units = units, use_bias = FALSE)
    self$wo <- layer_dense(units = units, use_bias = FALSE)

    self$num_heads <- num_heads
    self$head_size <- head_size
    self$block_id <- block_id
  },

  call = function(input) {
    # input = {batch_size, seq_len, n_features = head_size*num_heads}
    c(batch_size, seqlen, n_features) %<-% tf$unstack(tf$shape(input))

    # project x into query, key, value
    # self$wq(x) = {batch_size, seqlen, num_heads*head_size}
    split_heads_shape <- c(batch_size, seqlen, self$num_heads, self$head_size)
    q <- input |> self$wq() |> tf$reshape(split_heads_shape)
    k <- input |> self$wk() |> tf$reshape(split_heads_shape)
    v <- input |> self$wv() |> tf$reshape(split_heads_shape)

    # embed positional information in query and key
    # TODO

    # reshape
    # move heads out of last two axes so that matmuls
    # are performed across heads (seqlen, head_size) axes
    v <- tf$transpose(v, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen, head_size}
    q <- tf$transpose(q, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen, head_size}
    k <- tf$transpose(k, c(0L, 2L, 3L, 1L)) # {batch_size, num_heads, head_size, seqlen}

    # calculate attention scores
    scores <- tf$matmul(q, k)/tf$sqrt(1.0*self$head_size) # {batch_size, num_heads, seqlen, seqlen}
    mask <- make_mask(seqlen, dtype = scores$dtype) # {1, 1, seqlen, seqlen}
    scores <- scores + mask
    scores <- k_softmax(scores) # {batch_size, num_heads, seqlen, seqlen}

    # calculate value
    output <- tf$matmul(scores, v) # {batch_size, num_heads, seqlen, head_size}

    # combine heads back into single features dimension
    output <- output |>
      tf$transpose(c(0L, 2L, 1L, 3L)) |>  # {batch_size, seqlen, num_heads, head_size}
      tf$reshape(tf$shape(input)) # {batch_size, seqlen, num_heads*head_size}

    # final linear projection
    self$wo(output)
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("num_heads", "head_size", "block_id")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

layer_transformer_block <- new_layer_class(
  "TransformerBlock",
  initialize = function(num_heads, head_size, multiple_of, norm_eps = k_epsilon, ..., block_id = NULL) {
    super$initialize(...)

    # self attention
    self$sa <- layer_multi_self_attention(
      num_heads = num_heads,
      head_size = head_size,
      block_id = block_id)

    # feed forward
    self$ffwd <- layer_feedforward(
      hidden_dim = 4*head_size*num_heads,
      multiple_of = multiple_of,
      block_id = block_id)

    # layer norms
    self$sa_norm <- layer_rmsnorm(eps = norm_eps, block_id = block_id, feeds_into = "attention")
    self$ffwd_norm <- layer_rmsnorm(eps = norm_eps, block_id = block_id, feeds_into = "ffn")

    # config
    self$num_heads <- num_heads
    self$head_size <- head_size
    self$norm_eps <- norm_eps
    self$block_id <- block_id
  },

  call = function(input) {
    # residual connection
    x <- input + self$sa(self$sa_norm(input))
    x <- x + self$ffwd(self$ffwd_norm(x))
    x
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("num_heads", "head_size", "norm_eps", "block_id")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

# model
inputs <- layer_input(shape = shape(NA), name = "input")
token_embeddings <- inputs |>
  layer_embedding(
  input_dim = tokenizer$vocab_size(),
  output_dim = params$dim,
  name = "token_embeddings"
)

transformer_blocks <- token_embeddings
for (i in (seq_len(params$n_layers) - 1L)) {
  transformer_blocks <- transformer_blocks |>
    layer_transformer_block(
      num_heads = params$n_heads,
      head_size = params$dim %/% params$n_heads,
      multiple_of = params$multiple_of,
      norm_eps = params$norm_eps,
      block_id = i,
      name = paste0("transformer_", i))
}

output <- transformer_blocks |>
  layer_rmsnorm(block_id = -1, eps = params$norm_eps) |>
  layer_dense(units = tokenizer$vocab_size(), use_bias = FALSE, name = "linear_head")

model <- keras_model(inputs, output, name = "llama_model")
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
