rope_angles <- function(feature_dim, theta = 10000) {
  delta <- 1/(feature_dim %/% 2)
  freqs <- tf$range(start = 0, limit = 1, delta = delta, dtype = tf$float32)
  1.0/(theta^freqs) # {feature_dim/2}
}

#' Rotation Matrix for RoPE
#' @param seqlen size of sequence
#' @param feature_dim size of features
#' @export
rope_matrix <- function(seqlen, feature_dim, theta = 10000, dtype = keras::k_floatx()) {
  # vector of angles
  angles <- rope_angles(feature_dim, theta = theta) # {feature_dim/2}

  # vector representing position of tokens
  pos <- tf$range(seqlen, dtype = angles$dtype) # {seqlen}

  # outer product
  # out[i,j] <- pos[i]*angles[j]
  out <- tf$einsum('a,b->ab', pos, angles) # {seqlen, feature_dim/2}

  # broadcast
  out <- out[tf$newaxis, , tf$newaxis, ] # {1, seqlen, 1, feature_dim/2}

  # split into list and cast
  list(
    cos = out |> tf$cos() |> repeat_twice() |> tf$cast(dtype), # {1, seqlen, 1, feature_dim}
    sin = out |> tf$sin() |> repeat_twice() |> tf$cast(dtype)  # {1, seqlen, 1, feature_dim}
  )
}

rotate_every_two <- function(x) {
  x1 <- x[all_dims(), `1::2`]
  x2 <- x[all_dims(), `2::2`]
  y <- tf$stack(list(-x2, x1), axis = -1L)
  tf$reshape(y, tf$shape(x))
}

repeat_twice <- function(x) {
  tf$`repeat`(x, 2L, axis = -1L)
}

#' ROtational Position Embeddings
#' @param x inputs
#' @param rots precomputed rotational matrix from \code{rotation_matrix}
#' @export
rope <- function(x, rots) {
  # based on equation 34 from Roformers paper

  # x = {batch_size, seqlen, num_heads, head_size}
  # freqs = {1, max_seqlen, 1, head_size}
  seqlen <- tf$shape(x)[2]

  # {1, seqlen, 1, head_size}
  rot_cos <- rots$cos[,NA:seqlen,,]
  rot_sin <- rots$sin[,NA:seqlen,,]

  # {batch_size, seqlen, num_heads, head_size}
  y <- rotate_every_two(x)
  (x * rot_cos) + (y * rot_sin)
}
