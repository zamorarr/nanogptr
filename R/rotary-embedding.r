rotary_embedding <- function(x) {
  c(batch_size, seqlen, num_heads, head_size) %<-% tf$unstack(tf$shape(x))

  m <- rotation_matrix(seqlen, head_size) # {1, seqlen, 1, head_size/2}
  x <- view_as_complex(x) # {batch_size, seqlen, num_heads, head_size/2}
  x <- x * m # {batch_size, seqlen, num_heads, head_size/2}
  view_as_real(x) # {batch_size, seqlen, num_heads, head_size}
}

rotation_matrix <- function(seqlen, feature_dim, theta = 10000) {
  tk <- tf$range(seqlen, dtype = tf$float32) # {seqlen}

  delta <- 1/(feature_dim %/% 2)
  freqs <- tf$range(start = 0, limit = 1, delta = delta, dtype = tf$float32)
  freqs <- 1.0/(theta^freqs) # {feature_dim/2}

  # outer product
  freqs <- tf$einsum('a,b->ab', tk, freqs) # {seqlen, feature_dim/2}
  mat <- tf$complex(tf$cos(freqs), tf$sin(freqs)) # {seqlen, feature_dim/2}

  # broadcast
  mat[tf$newaxis, , tf$newaxis, ] # {1, seqlen, 1, feature_dim/2}
}

view_as_complex <- function(x) {
  # x = {batch_size, seqlen, num_heads, head_size}

  # get every 2nd item from the last dimension
  a <- x[all_dims(),`::2`] # {batch_size, seqlen, num_heads, head_size/2}
  b <- x[all_dims(), `2::2`] # {batch_size, seqlen, num_heads, head_size/2 - 1}
  tf$complex(a, b)
}

view_as_real <- function(x) {

}
