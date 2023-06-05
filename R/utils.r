gpt_params <- function(block_size, vocab_size, n_layer, n_head, n_embd) {
  out <- list()

  # embeddings
  out[["embedding/position"]] <- n_embd * block_size
  out[["embedding/token"]] <- n_embd * vocab_size
  out[["embedding"]] <- out[["embedding/position"]] + out[["embedding/token"]]

  # attention blocks
  out[["attention/ln"]] <- n_embd
  out[["attention/kqv"]] <- n_embd * 3*n_embd
  out[["attention/proj"]] <- n_embd^2
  out[["attention"]] <- out[["attention/ln"]] + out[["attention/kqv"]] + out[["attention/proj"]]

  # MLP blocks
  ffw_size <- 4*n_embd
  out[["mlp/ln"]] <- n_embd
  out[["mlp/ffw"]] <- n_embd * ffw_size
  out[["mlp/proj"]] <- ffw_size * n_embd
  out[["mlp"]] <- out[["mlp/ln"]] + out[["mlp/ffw"]] + out[["mlp/proj"]]

  # transformer
  out[["block"]] <- out[["attention"]] + out[["mlp"]]
  out[["transformers"]] <- n_layer * out[["block"]]

  # model head
  out[["ln_f"]] <- n_embd
  out[["dense"]] <- 0 #out[["embedding"]] # 0 if layer sharing with embedding/position

  # total
  out$total <- out$embedding + out$transformer + out$ln_f + out$dense

  data.frame(name = names(out), params = unname(unlist(out)))
}

llama_params <- function(vocab_size, n_layer, n_head, n_embd, multiple_of) {
  out <- list()

  # embeddings
  out[["embedding/token"]] <- n_embd * vocab_size
  out[["embedding"]] <- out[["embedding/token"]]

  # attention blocks
  out[["attention/rmsnorm"]] <- n_embd
  out[["attention/kqvo"]] <- n_embd * 4*n_embd
  # out[["attention/rope"]] <- ?
  out[["attention"]] <- out[["attention/rmsnorm"]] + out[["attention/kqvo"]]

  # MLP blocks
  ffw_size <- round(2/3*4*n_embd/multiple_of)*multiple_of
  out[["mlp/rmsnorm"]] <- n_embd
  out[["mlp/w1"]] <- n_embd * ffw_size
  out[["mlp/w2"]] <- ffw_size * n_embd
  out[["mlp/w3"]] <- n_embd * ffw_size
  out[["mlp"]] <- out[["mlp/rmsnorm"]] + out[["mlp/w1"]] + out[["mlp/w2"]] + out[["mlp/w3"]]

  # transformer
  out[["block"]] <- out[["attention"]] + out[["mlp"]]
  out[["transformers"]] <- n_layer * out[["block"]]

  # model head
  out[["rmsnorm_f"]] <- n_embd
  out[["dense"]] <- out[["embedding"]] # 0 if layer sharing with embedding/position

  # total
  out$total <- out$embedding + out$transformer + out$rmsnorm_f + out$dense

  df <- data.frame(name = names(out), params = unname(unlist(out)))
  df$ratio <- sprintf("%10.4f", 100*df$params/out$total)
  df
}

#llama_params(tokenizer$vocab_size()$numpy(), params$n_layers, params$n_heads, params$dim, params$multiple_of)
llama_params(32000, 32, 32, 4096, 256)
#model$count_params()
