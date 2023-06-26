#' Load LLaMA tokenizer
#' @param path to tokenizer.model file
#' @export
llama_tokenizer <- function(path) {
  tf_text <- reticulate::import("tensorflow_text")
  path <- normalizePath(path, mustWork = TRUE)

  tokenizer <-tf$io$gfile$GFile(path, "rb")$read()
  tf_text$SentencepieceTokenizer(tokenizer, add_bos = TRUE, add_eos = FALSE)
}

# custom layers

#' LLaMA RMS Norm Layer
layer_llama_rmsnorm <- keras::new_layer_class(
  "LlamaRMSNorm",
  initialize = function(eps = 1E-6, ...) {
    super$initialize(...)
    self$eps <- eps
  },

  build = function(input_shape) {
    # input_shape = (batch_size, seqlen, model_dim)
    # w_shape = (1, 1, model_dim)
    size <- length(input_shape)
    model_dim <- tf$constant(tail(input_shape, 1), shape = keras::shape(1))
    paddings <- tf$constant(array(c(size - 1L, 0L), dim = c(1, 2)))
    w_shape = tf$pad(model_dim, paddings, constant_values = 1L)
    self$w <- self$add_weight(
      shape = w_shape,
      dtype = tf$dtypes$float32,
      initializer = "ones",
      trainable = TRUE,
      name = "kernel")
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
    input_dtype <- x$dtype
    # cast x to float32 first to maintain precision
    x32 <- tf$cast(x, dtype = tf$dtypes$float32)
    result <- x32 * self$rrms(x32) * self$w

    # cast result back to input dtype
    tf$cast(result, dtype = input_dtype)
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("eps")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

layer_llama_feedforward <- keras::new_layer_class(
  "LlamaFeedForward",
  initialize = function(output_dim, multiple_of = 256L, ...) {
    super$initialize(...)

    # compute hidden_dim
    hidden_dim <- as.integer(4 * output_dim * (2/3))
    hidden_dim <- (hidden_dim + multiple_of - 1) %/% multiple_of
    hidden_dim <- hidden_dim * multiple_of

    # dense layers
    self$w1 <- keras::layer_dense(units = hidden_dim, use_bias = FALSE, name = "w1")
    self$w2 <- keras::layer_dense(units = output_dim, use_bias = FALSE, name = "w2")
    self$w3 <- keras::layer_dense(units = hidden_dim, use_bias = FALSE, name = "w3")

    # config
    self$output_dim <- output_dim
  },

  call = function(x) {
    # SwiGLU
    self$w2(tf$nn$silu(self$w1(x)) * self$w3(x))
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("output_dim")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

llama_make_mask <- function(seqlen, dtype) {
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

layer_llama_multi_self_attention <- keras::new_layer_class(
  "LlamaMultiSelfAttention",
  initialize = function(num_heads, head_size, ...) {
    super$initialize(...)

    # layers
    units <- num_heads * head_size
    self$wq <- keras::layer_dense(units = units, use_bias = FALSE, name = "wq")
    self$wk <- keras::layer_dense(units = units, use_bias = FALSE, name = "wk")
    self$wv <- keras::layer_dense(units = units, use_bias = FALSE, name = "wv")
    self$wo <- keras::layer_dense(units = units, use_bias = FALSE, name = "wo")

    self$num_heads <- num_heads
    self$head_size <- head_size
    self$score_norm <- tf$sqrt(tf$convert_to_tensor(head_size, dtype = keras::k_floatx()))
  },

  call = function(input, rots = NULL) {
    # input = {batch_size, seq_len, n_features = head_size*num_heads}
    c(batch_size, seqlen, n_features) %<-% tf$unstack(tf$shape(input))

    # project x into query, key, value
    # self$wq(x) = {batch_size, seqlen, num_heads*head_size}
    split_heads_shape <- c(batch_size, seqlen, self$num_heads, self$head_size)
    q <- input |> self$wq() |> tf$reshape(split_heads_shape)
    k <- input |> self$wk() |> tf$reshape(split_heads_shape)
    v <- input |> self$wv() |> tf$reshape(split_heads_shape)

    # embed positional information in query and key
    if (is.null(rots)) rots <- rope_matrix(seqlen, self$head_size)
    q <- rope(q, rots)
    k <- rope(k, rots)

    # reshape
    # move heads out of last two axes so that matmuls
    # are performed across heads (seqlen, head_size) axes
    v <- tf$transpose(v, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen, head_size}
    q <- tf$transpose(q, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen, head_size}
    k <- tf$transpose(k, c(0L, 2L, 3L, 1L)) # {batch_size, num_heads, head_size, seqlen}

    # calculate attention scores
    scores <- tf$matmul(q, k)/self$score_norm # {batch_size, num_heads, seqlen, seqlen}
    mask <- llama_make_mask(seqlen, dtype = scores$dtype) # {1, 1, seqlen, seqlen}
    scores <- scores + mask

    # softmax should be fp32
    scores <- scores |>
      tf$cast(tf$dtypes$float32) |>
      tf$nn$softmax() |>
      tf$cast(q$dtype) # {batch_size, num_heads, seqlen, seqlen}
      #keras::k_softmax(scores)

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
    params <- c("num_heads", "head_size")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

layer_llama_transformer_block <- keras::new_layer_class(
  "LlamaTransformerBlock",
  initialize = function(num_heads, head_size, multiple_of, norm_eps = keras::k_epsilon(), ...) {
    super$initialize(...)

    # self attention
    self$sa <- layer_llama_multi_self_attention(
      name = "attention",
      num_heads = num_heads,
      head_size = head_size)

    # feed forward
    self$ffwd <- layer_llama_feedforward(
      name = "feed_forward",
      output_dim = head_size*num_heads,
      multiple_of = multiple_of)

    # layer norms
    self$sa_norm <- layer_llama_rmsnorm(eps = norm_eps, name = "attention_norm")
    self$ffwd_norm <- layer_llama_rmsnorm(eps = norm_eps, name = "ffn_norm")

    # config
    self$num_heads <- num_heads
    self$head_size <- head_size
    self$norm_eps <- norm_eps
  },

  call = function(input, rots = NULL) {
    # residual connection
    x <- input + self$sa(self$sa_norm(input), rots)
    x <- x + self$ffwd(self$ffwd_norm(x))
    x
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("num_heads", "head_size", "norm_eps")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

#' Create LLaMA model
#' @param params list of params
#' @export
llama_model <- function(params) {
  # compute rotational embedding frequencies
  max_context_size <- 2048L
  head_size <- params$dim %/% params$n_heads
  rots <- rope_matrix(max_context_size,  head_size)

  inputs <- keras::layer_input(shape = keras::shape(NA), name = "input")

  token_embeddings <- inputs |>
    keras::layer_embedding(
      input_dim = params$vocab_size,
      output_dim = params$dim,
      name = "tok_embeddings"
    )

  transformer_blocks <- token_embeddings
  for (i in (seq_len(params$n_layers) - 1L)) {
    block <- layer_llama_transformer_block(
      num_heads = params$n_heads,
      head_size =  head_size,
      multiple_of = params$multiple_of,
      norm_eps = params$norm_eps,
      name = paste0("transformer_", i))
    transformer_blocks <- block(transformer_blocks, rots = rots)
  }

  output <- transformer_blocks |>
    layer_llama_rmsnorm(eps = params$norm_eps, name = "norm") |>
    keras::layer_dense(units = params$vocab_size, use_bias = FALSE, name = "output")

  keras::keras_model(inputs, output, name = "llama_model")
}

llama_load_weights <- function(model, params, path = "~/data/llama-np/7B", from_hf = FALSE) {
  np <- reticulate::import("numpy", convert = FALSE)
  # all layer names
  #lapply(model$layers, \(layer) vapply(layer$weights, \(x) x$name, character(1)))
  #lapply(model$weights, \(w) w$name)

  # name map
  layermap <- readr::read_csv("inst/examples/llama-7b-layers.csv")
  layermap <- setNames(layermap$llama_name, layermap$keras_name)
  #layermap

  for (layer in model$layers) {
    for (weight in layer$weights) {
      cat("loading", weight$name, "with shape", paste(tf$shape(weight)$numpy(), collapse="x"), " ")

      stopifnot(weight$name %in% names(layermap))
      llama_name <- layermap[[weight$name]]
      value <- llama_load_weight(path, llama_name)

      if (grepl("attention/w[qk]/kernel:0", weight$name)) {
        if (from_hf) {
          message("unpermuting hf")
          # have to unpermute first
          n_heads <- params$n_heads
          mdim <- params$dim
          head_size <- mdim %/% n_heads
          value <- value |>
            np$reshape(as.integer(c(n_heads, 2, head_size %/% 2, mdim))) |>
            np$transpose(c(0L, 2L, 1L, 3L)) |>
            np$reshape(c(mdim, mdim))
        }
        value <- np$transpose(value)
      } else if (grepl("attention/w[vo]/kernel:0", weight$name)) {
        value <- np$transpose(value)
      } else if (grepl("forward/w[1-3]/kernel:0", weight$name)) {
        value <- np$transpose(value)
      } else if (grepl("norm/kernel:0", weight$name)) {
        value <- np$expand_dims(value, c(0L, 1L))
      } else if (grepl("output/kernel:0", weight$name)) {
        value <- np$transpose(value)
      }

      cat("from values with shape", paste(value$shape, collapse="x"), "\n")
      weight$assign(value)
    }
  }

  invisible(model)
}

llama_load_weight <- function(path, name) {
  np <- reticulate::import("numpy", convert = FALSE)

  #llama_name <- layermap[[keras_name]]
  path <- file.path(path, sprintf("%s.npy", name))
  path <- normalizePath(path, mustWork = TRUE)
  np$load(path)
}
