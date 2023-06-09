# rotary_embedding <- function(x) {
#   c(batch_size, seqlen, num_heads, head_size) %<-% tf$unstack(tf$shape(x))
#
#   m <- rotation_matrix(seqlen, head_size) # {1, seqlen, 1, head_size/2}
#   x <- view_as_complex(x) # {batch_size, seqlen, num_heads, head_size/2}
#   x <- x * m # {batch_size, seqlen, num_heads, head_size/2}
#   view_as_real(x) # {batch_size, seqlen, num_heads, head_size}
# }

rotation_freqs <- function(seqlen, feature_dim, theta = 10000) {
  # number representing position of tokens
  tk <- tf$range(seqlen, dtype = tf$float32) # {seqlen}

  # array of angles
  delta <- 1/(feature_dim %/% 2)
  freqs <- tf$range(start = 0, limit = 1, delta = delta, dtype = tf$float32)
  freqs <- 1.0/(theta^freqs) # {feature_dim/2}

  # outer product
  # out[i,j] <- tk[i]*freqs[j]
  out <- tf$einsum('a,b->ab', tk, freqs) # {seqlen, feature_dim/2}
  mat <- tf$complex(tf$cos(out), tf$sin(out)) # {seqlen, feature_dim/2}

  # broadcast
  mat <- mat[tf$newaxis, , tf$newaxis, ] # {1, seqlen, 1, feature_dim/2}

  # split into list
  list(
    cos = tf$math$real(mat) |> tf$`repeat`(2L, axis = -1L), # {1, seqlen, 1, feature_dim}
    sin = tf$math$imag(mat) |> tf$`repeat`(2L, axis = -1L) # {1, seqlen, 1, feature_dim}
  )
}

# view_as_complex <- function(x) {
#   # x = {batch_size, seqlen, num_heads, head_size}
#
#   # combine subsequent indices into pairs of a complex number
#   # tensorflow slicing with R. see ?`[.tensorflow.tensor`
#   a <- x[all_dims(),`1::2`] # {batch_size, seqlen, num_heads, head_size/2}
#   b <- x[all_dims(), `2::2`] # {batch_size, seqlen, num_heads, head_size/2 - 1}
#   tf$complex(a, b)
# }
#
# view_as_real <- function(x) {
#
# }

rotate_every_two <- function(x) {
  x1 <- x[all_dims(), `1::2`]
  x2 <- x[all_dims(), `2::2`]
  y <- tf$stack(list(-x2, x1), axis = -1L)
  tf$reshape(y, tf$shape(x))
}

rotate_eff <- function(x, freqs) {
  # based on equation 34 from Roformers paper

  # x = {batch_size, seqlen, num_heads, head_size}
  # freqs = {1, max_seqlen, 1, head_size}
  seqlen <- tf$shape(x)[2]

  # {1, seqlen, 1, head_size}
  rot_cos <- freqs$cos[,NA:seqlen,,]
  rot_sin <- freqs$sin[,NA:seqlen,,]

  # {batch_size, seqlen, num_heads, head_size}
  y <- rotate_every_two(x)
  (x * rot_cos) + (y * rot_sin)
}
