prettify_size <- function(df) {
  #df$ratio <- sprintf("%10.4f", 100*df$params/out$total)
  bytes <- df$params*4 # fp32 = 4 bytes per param
  bytes_and_buffers <- bytes + 2*bytes # adam optimizer stores an additional 2 buffers per param
  df$mem_inference <- prettyunits::pretty_bytes(bytes)
  df$mem_train <- prettyunits::pretty_bytes(bytes_and_buffers)

  formatter <- scales::label_number(0.1, scale_cut = scales::cut_short_scale())
  df$params <- formatter(df$params)
  df
}

#' Estimate GPT Total Parameters
#' @param vocab_size number of vocabulary tokens
#' @param n_layer number of transformer layers
#' @param n_head number of transfomer heads per layer
#' @param n_embed size of feature (embedding) dimension
#' @param block_size max context size for position encodings
#' @examples
#' gpt_size(50257, 12, 12, 768, 1024) # gpt2
#' gpt_size(50257, 24, 16, 1024, 1024) # gpt2-med
#' gpt_size(50257, 36, 20, 1280, 1024) # gpt2-large
#' gpt_size(50257, 48, 25, 1600, 1024) # gpt2-xl
#'
#' @export
gpt_size <- function(vocab_size, n_layer, n_head, n_embed, block_size) {
  out <- list()

  # embeddings
  out[["embedding/position"]] <- n_embed * block_size
  out[["embedding/token"]] <- n_embed * vocab_size
  out[["embedding"]] <- out[["embedding/position"]] + out[["embedding/token"]]

  # attention blocks
  out[["attention/ln"]] <- n_embed
  out[["attention/kqv"]] <- n_embed * 3*n_embed
  out[["attention/proj"]] <- n_embed^2
  out[["attention"]] <- out[["attention/ln"]] + out[["attention/kqv"]] + out[["attention/proj"]]

  # MLP blocks
  ffw_size <- 4*n_embed
  out[["mlp/ln"]] <- n_embed
  out[["mlp/ffw"]] <- n_embed * ffw_size
  out[["mlp/proj"]] <- ffw_size * n_embed
  out[["mlp"]] <- out[["mlp/ln"]] + out[["mlp/ffw"]] + out[["mlp/proj"]]

  # transformer
  out[["transformers/block"]] <- out[["attention"]] + out[["mlp"]]
  out[["transformers"]] <- n_layer * out[["transformers/block"]]

  # model head
  out[["final/ln"]] <- n_embed
  out[["final/dense"]] <- 0 #out[["embedding"]] # 0 if layer sharing with embedding/position
  out[["final"]] <- out[["final/ln"]] + out[["final/dense"]]

  # total
  out$total <- out$embedding + out$transformers + out$final

  df <- data.frame(name = names(out), params = unname(unlist(out)))
  prettify_size(df)
}

llama_params <- function(vocab_size, n_layer, n_head, n_embed, multiple_of = 256L, norm_eps = 1E-6) {
  list(
    dim = n_embed,
    multiple_of = multiple_of,
    n_heads = n_head,
    n_layers = n_layer,
    norm_eps = norm_eps,
    vocab_size = vocab_size
  )
}

#' Estimate LLaMA Total Parameters
#' @param vocab_size number of vocabulary tokens
#' @param n_layer number of transformer layers
#' @param n_head number of transfomer heads per layer
#' @param n_embed size of feature (embedding) dimension
#' @param multiple_of number to round feedfoward dimension to
#' @examples
#' llama_size(32000, 32, 32, 4096, 256) # llama-7B
#' llama_size(32000, 40, 40, 5120, 256) # llama-13B
#' llama_size(32000, 60, 52, 6656, 256) # llama-32.5B
#' llama_size(32000, 80, 64, 8192, 256) # llama-65.2B
#' @export
llama_size <- function(vocab_size, n_layer, n_head, n_embed, multiple_of = 256L, pretty = TRUE) {
  out <- list()

  # embeddings
  out[["embedding/token"]] <- n_embed * vocab_size
  out[["embedding"]] <- out[["embedding/token"]]

  # attention blocks
  out[["attention/rmsnorm"]] <- n_embed
  out[["attention/kqvo"]] <- n_embed * 4*n_embed
  out[["attention"]] <- out[["attention/rmsnorm"]] + out[["attention/kqvo"]]

  # MLP blocks
  ffw_size <- round(2/3*4*n_embed/multiple_of)*multiple_of
  out[["mlp/rmsnorm"]] <- n_embed
  out[["mlp/w1"]] <- n_embed * ffw_size
  out[["mlp/w2"]] <- ffw_size * n_embed
  out[["mlp/w3"]] <- n_embed * ffw_size
  out[["mlp"]] <- out[["mlp/rmsnorm"]] + out[["mlp/w1"]] + out[["mlp/w2"]] + out[["mlp/w3"]]

  # transformer
  out[["transformers/block"]] <- out[["attention"]] + out[["mlp"]]
  out[["transformers"]] <- n_layer * out[["transformers/block"]]

  # model head
  out[["final/rmsnorm"]] <- n_embed
  out[["final/dense"]] <- out[["embedding/token"]] # 0 if layer sharing with embedding/position
  out[["final"]] <- out[["final/rmsnorm"]] + out[["final/dense"]]

  # total
  out$total <- out$embedding + out$transformers + out$final

  df <- data.frame(name = names(out), params = unname(unlist(out)))
  if (pretty) df <- prettify_size(df)
  df
}

#llama_params(tokenizer$vocab_size()$numpy(), params$n_layers, params$n_heads, params$dim, params$multiple_of)
#model$count_params()
