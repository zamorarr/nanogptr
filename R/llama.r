#' Load LLaMA tokenizer
#' @param path to tokenizer.model file
#' @export
llama_tokenizer <- function(path) {
  # could not install tf_text on macos
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

  rrms = function(x, eps) {
    # reciprocal root mean square along last axis
    # same as dividing by std dev? assuming zero mean
    # x = {batch_size, seqlen, model_dim}
    x |>
      tf$math$square() |> # {B, T, D}
      tf$reduce_mean(axis = -1L, keepdims = TRUE) |> # {B, T, 1}
      tf$math$add(eps) |> # {B, T, 1}
      tf$math$rsqrt() # {B, T, 1}
  },

  call = function(x) {
    input_dtype <- x$dtype
    # cast x to float32 first to maintain precision
    x32 <- tf$cast(x, dtype = tf$dtypes$float32)
    result <- x32 * self$rrms(x32, eps = self$eps) * self$w

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

#' Calculate hidden dim for feedforward network
#'
#' Rounds 8/3*output_dim to the next multiple of multiple_of
llama_hidden_dim <- function(output_dim, multiple_of) {
  hidden_dim <- as.integer(4 * output_dim * (2/3))
  hidden_dim <- (hidden_dim + multiple_of - 1) %/% multiple_of
  hidden_dim <- hidden_dim * multiple_of
  hidden_dim
}

layer_llama_feedforward <- keras::new_layer_class(
  "LlamaFeedForward",
  initialize = function(output_dim, multiple_of = 256L, ...) {
    super$initialize(...)

    # compute hidden_dim
    hidden_dim <- llama_hidden_dim(output_dim, multiple_of)

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
  y <- tf$range(seqlen)
  row_index <- x[, tf$newaxis]
  col_index <- y[tf$newaxis, ]
  mask <- tf$where(
    row_index < col_index,
    tf$constant(-Inf, dtype = dtype),
    tf$constant(0, dtype = dtype)
  )

  # {1, 1, seqlen_q, seqlen_kv}
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

  # rots = rotation_matrix
  # mask = casual mask = {1, 1, seqlen_q, seqlen_kv}
  # cache = list of (k/v cached tensors)
  # cache_pos = current position in seqlen we are predicting (using cache)
  call = function(input, attn_rots, attn_mask, training = FALSE, cache = NULL, cache_pos = 0L) {
    if (training && !is.null(cache)) stop("cannot use k/v cache while in training mode", call. = FALSE)

    # input = {batch_size, seqlen, n_features = head_size*num_heads}
    c(batch_size, seqlen_q, n_features) %<-% tf$unstack(tf$shape(input))

    # project x into query, key, value
    # self$wq(x) = {batch_size, seqlen_q, num_heads*head_size}
    split_heads_shape <- c(batch_size, seqlen_q, self$num_heads, self$head_size)
    q <- input |> self$wq() |> tf$reshape(split_heads_shape)
    k <- input |> self$wk() |> tf$reshape(split_heads_shape)
    v <- input |> self$wv() |> tf$reshape(split_heads_shape)

    # embed positional information in query and key
    q <- rope(q, attn_rots)
    k <- rope(k, attn_rots)

    # use cache for k/v vectors
    # see: https://keras.io/api/keras_nlp/modeling_layers/cached_multi_head_attention/
    # https://github.com/keras-team/keras-nlp/blob/v0.5.1/keras_nlp/layers/cached_multi_head_attention.py#L23
    if (!is.null(cache)) {
      # unpack cache
      c(cache_k, cache_v) %<-% keras::k_unstack(cache, axis = 2L)

      # prepend caches to current k,v
      update_slice <- tf$compiler$tf2xla$python$xla$dynamic_update_slice
      cache_start <- c(0L, cache_pos, 0L, 0L)
      cache_k <- update_slice(cache_k, k, cache_start)
      cache_v <- update_slice(cache_v, v, cache_start)

      # need to trim k/v back to seqlen. cache is max_seqlen
      seqlen_kv <- seqlen_q + cache_pos
      k <- cache_k[, 1:seqlen_kv, ,]
      v <- cache_v[, 1:seqlen_kv, ,]

      # pack cache
      cache <- keras::k_stack(list(cache_k, cache_v), axis = 2L)
    }

    # reshape
    # move heads out of last two axes so that matmuls
    # are performed across heads (seqlen, head_size) axes
    q <- tf$transpose(q, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen_q, head_size}
    k <- tf$transpose(k, c(0L, 2L, 3L, 1L)) # {batch_size, num_heads, head_size, seqlen_kv}
    v <- tf$transpose(v, c(0L, 2L, 1L, 3L)) # {batch_size, num_heads, seqlen_kv, head_size}

    # calculate attention scores
    scores <- tf$matmul(q, k)/self$score_norm # {batch_size, num_heads, seqlen_q, seqlen_kv}
    scores <- scores + attn_mask

    # softmax should be fp32
    scores <- scores |>
      tf$cast(tf$dtypes$float32) |>
      tf$nn$softmax() |>
      tf$cast(q$dtype) # {batch_size, num_heads, seqlen_q, seqlen_kv}

    # calculate value
    output <- tf$matmul(scores, v) # {batch_size, num_heads, seqlen_q, head_size}

    # combine heads back into single features dimension
    output <- output |>
      tf$transpose(c(0L, 2L, 1L, 3L)) |>  # {batch_size, seqlen_q, num_heads, head_size}
      tf$reshape(tf$shape(input)) # {batch_size, seqlen_q, num_heads*head_size}

    # final linear projection
    output <- self$wo(output) # {batch_size, seqlen_q, model_dim = num_heads*head_size}
    list(output, cache)
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

  call = function(input, attn_rots, attn_mask, ...) {
    # trim mask
    c(batch_size, seqlen, n_features) %<-% tf$unstack(tf$shape(input))
    #attn_mask <- attn_mask[1,1,1:seqlen,1:seqlen, drop = FALSE] # does this work with cache?

    # self-attention + residual
    residual <- input
    values <- input |>
      self$sa_norm() |>
      self$sa(attn_rots = attn_rots, attn_mask = attn_mask, ...)
    x <- values[[1]]
    cache <- values[[2]]
    x <- x + residual

    # ffwd + residual
    residual <- x
    x <- x |>
      self$ffwd_norm() |>
      self$ffwd()
    x <- x + residual

    # return output and cache
    list(x, cache)
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
llama_model <- function(params, max_seqlen = 2048L, use_cache = FALSE) {
  # compute rotational embedding frequencies
  head_size <- params$dim %/% params$n_heads
  attn_rots <- rope_matrix(max_seqlen, head_size)

  # compute mask
  attn_mask <- llama_make_mask(max_seqlen, dtype = tf$dtypes$float32)

  # layers
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
    c(transformer_blocks, cache) %<-% block(transformer_blocks, attn_rots = attn_rots, attn_mask = attn_mask)
  }

  output <- transformer_blocks |>
    layer_llama_rmsnorm(eps = params$norm_eps, name = "norm") |>
    keras::layer_dense(units = params$vocab_size, use_bias = FALSE, name = "output")

  keras::keras_model(inputs, output, name = "llama_model")
}

llama_model2 <- keras::new_model_class(
  "LlamaModel",
  initialize = function(params, max_seqlen = 2048L, ...) {
    super$initialize(...)
    self$params <- params

    head_size <-  params$dim %/% params$n_heads

    # pre-compute max-size rotation matrix
    self$attn_rots <- rope_matrix(max_seqlen, head_size)

    # pre-compute max-size mask
    self$attn_mask <- llama_make_mask(max_seqlen, keras::k_floatx())

    # embedding layers
    self$tok_embeddings <- keras::layer_embedding(
      input_dim = params$vocab_size,
      output_dim = params$dim,
      name = "tok_embeddings")

    # transformer blocks
    self$transformer_blocks <- lapply(seq(params$n_layers), function(i) {
      layer_llama_transformer_block(
        num_heads = params$n_heads,
        head_size = head_size,
        multiple_of = params$multiple_of,
        norm_eps = params$norm_eps,
        name = paste0("transformer_", i - 1L))
    })

    # final layers
    self$norm <- layer_llama_rmsnorm(eps = params$norm_eps, name = "norm")
    self$lm_head <- layer_dense(units = params$vocab_size, use_bias = FALSE, name = "output")
  },

  # the input here should already be padded to the max seqlen
  create_cache = function(input) {
    # input = {batch_size, max_seqlen}
    params <- self$params
    head_size <-  params$dim %/% params$n_heads

    # get input size
    c(batch_size, max_seqlen) %<-% keras::k_unstack(keras::k_shape(input))

    # cache = {num_layers, batch_size, 2, max_seqlen, num_heads, head_size}
    cache_shape <- c(params$n_layers, batch_size, 2L, max_seqlen, params$n_heads, head_size)
    tf$zeros(cache_shape, dtype = "float32")
  },

  call = function(input, cache = NULL, cache_pos = 0L) {
    # get input shape
    c(batch_size, seqlen) %<-% tf$unstack(tf$shape(input))

    # seqlen-size mask and rotation matrix
    mask <- self$attn_mask[, , 1:seqlen, 1:seqlen]

    # what happens when seqlen ==1? rots should be [,,cache_pos,,]?
    #zero <- tf$constant(0L)
    tfautograph::autograph({
      if (cache_pos > 0L) {
        rots <- self$attn_rots[, , cache_pos - 1L, ,] # need to adjust rotation to current token position
        rots <- rots[,,tf$newaxis,,]
      } else {
        rots <- self$attn_rots[, , 1:seqlen, ,]
      }
    })

    # custom forward pass
    x <- input |> self$tok_embeddings()

    if (is.null(cache)) {
      # do not use or save cache
      for (i  in seq_along(self$transformer_blocks)) {
        block <- self$transformer_blocks[[i]]
        c(x, cache_i) %<-% block(x, attn_rots = rots, attn_mask = mask, cache = NULL, cache_pos = cache_pos)
      }
    } else {
      # use and save cache
      cache <- keras::k_unstack(cache, axis = 1L)
      for (i  in seq_along(self$transformer_blocks)) {
        block <- self$transformer_blocks[[i]]
        c(x, cache_i) %<-% block(x, attn_rots = rots, attn_mask = mask, cache = cache[[i]], cache_pos = cache_pos)
        cache[[i]] <- cache_i
      }
      cache <- keras::k_stack(cache, axis = 1L)
    }

    output <- x |>
      self$norm() |> # rmsnorm
      self$lm_head() # dense projection

    # return output
    list(output = output, cache = cache)
  },

  get_config = function() {
    config <- super$get_config()
    params <- c("params")
    for (p in params) {
      config[[p]] <- self[[p]]
    }
    config
  }
)

llama_generate <- function(model, input, max_new_tokens = 10L, block_size = 64L) {
  # get input shape
  c(batch_size, input_seqlen) %<-% keras::k_unstack(keras::k_shape(input))

  # calculate max seqlen
  max_seqlen <- min(input_seqlen + max_new_tokens, block_size)
  #max_seqlen <- input_seqlen + max_new_tokens

  # XLA update tensor in place
  update_slice <- tf$compiler$tf2xla$python$xla$dynamic_update_slice

  # create output placeholder
  output <- tf$zeros(c(batch_size, max_seqlen), dtype = input$dtype)
  output <- update_slice(output, input, c(0L, 0L))

  # create cache
  cache <- model$create_cache(output)

  # define sampler
  #sampler <- function(x) tf$random$categorical(x, 1L, dtype = tf$dtypes$int32)
  sampler <- function(x) tf$expand_dims(tf$argmax(x, axis = 1L, output_type = input$dtype), axis = 0L)
  #sampler <- function(x) {
  #  x |>
  #    keras::k_argmax(axis = 2L) |>
  #    keras::k_cast("int32") |>
  #    keras::k_expand_dims(axis = 1L)
  #}

  # seed cache (forward pass)
  c(logits, cache) %<-% model(input, cache = cache, cache_pos = 0L)
  logits <- logits[,-1L,]
  idx_next <- sampler(logits)
  output <- update_slice(output, idx_next, c(0L, input_seqlen))

  # loop over next tokens
  #cache_pos <- 0L
  cache_pos <- tf$constant(0L, dtype = "int32")
  tfautograph::autograph({
    for (cache_pos in tf$range(input_seqlen + 1L, max_seqlen)) {
      # get the predictions
      c(logits, cache) %<-% model(idx_next, cache = cache, cache_pos = cache_pos) # {B, T, C}

      # focus on last context token
      logits <- logits[,-1L,]

      # sample from distribution
      idx_next <- sampler(logits) # {B, 1}

      # append sampled index to running sequence
      output <- update_slice(output, idx_next, c(0L, cache_pos)) # {B, T+1}
    }
  })

  # return final
  output
}

llama_generate_naive <- function(model, input, max_new_tokens = 10L, block_size = 64L) {
  # get input shape
  c(batch_size, input_seqlen) %<-% keras::k_unstack(keras::k_shape(input))

  # calculate max seqlen
  max_seqlen <- min(input_seqlen + max_new_tokens, block_size)

  # XLA update tensor in place
  update_slice <- tf$compiler$tf2xla$python$xla$dynamic_update_slice

  # create output placeholder
  output <- tf$zeros(c(batch_size, max_seqlen), dtype = input$dtype)
  output <- update_slice(output, input, c(0L, 0L))

  # define sampler
  #sampler <- function(x) tf$random$categorical(x, 1L, dtype = tf$dtypes$int32)
  sampler <- function(x) {
    x |>
      keras::k_argmax(axis = 2L) |>
      keras::k_cast("int32") |>
      keras::k_expand_dims(axis = 1L)
  }


  # loop over next tokens
  tfautograph::autograph({
    for (i in tf$range(input_seqlen, max_seqlen)) {
      # crop input
      input_cropped <- output[,1:i]

      # get the predictions
      c(logits, cache) %<-% model(input_cropped) # {B, T, C}

      # focus on last context token
      logits <- logits[,-1L,]

      # sample from distribution
      idx_next <- sampler(logits) # {B, 1}

      # append sampled index to running sequence
      output <- update_slice(output, idx_next, c(0L, i)) # {B, T+1}
    }
  })

  # return final
  output
}

llama_load_weights <- function(model, params, path = "~/data/llama-np/7B", from_hf = FALSE) {
  np <- reticulate::import("numpy", convert = FALSE)
  # all layer names
  #lapply(model$layers, \(layer) vapply(layer$weights, \(x) x$name, character(1)))
  #lapply(model$weights, \(w) w$name)

  # name map
  layermap <- readr::read_csv("inst/examples/llama-7b-layers.csv")
  layermap <- stats::setNames(layermap$llama_name, layermap$keras_name)
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
