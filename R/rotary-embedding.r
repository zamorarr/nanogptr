rope_angles <- function(feature_dim, theta = 10000) {
  n <- feature_dim %/% 2
  freqs <- keras::k_arange(0, n, dtype = tf$float32)/n
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
  pos <- keras::k_arange(seqlen, dtype = angles$dtype) # {seqlen}

  # outer product
  # out[i,j] <- pos[i]*angles[j]
  # {seqlen, 1} x {1 x feature_dim/2} = {seqlen, feature_dim/2}
  #out <- tf$einsum('a,b->ab', pos, angles) # {seqlen, feature_dim/2}
  out <- keras::k_dot(keras::k_expand_dims(pos), keras::k_expand_dims(angles, axis = 1))

  # broadcast
  out <- out[tf$newaxis, , tf$newaxis, ] # {1, seqlen, 1, feature_dim/2}

  # split into list and cast
  cos_out <- keras::k_cos(out) |> repeat_twice() |> keras::k_cast(dtype) # {1, seqlen, 1, feature_dim}
  sin_out <- keras::k_sin(out) |> repeat_twice() |> keras::k_cast(dtype)  # {1, seqlen, 1, feature_dim}
  keras::k_stack(list(cos_out, sin_out), axis = 1L)
}

rotate_every_two <- function(x) {
  x1 <- x[tensorflow::all_dims(), `1::2`]
  x2 <- x[tensorflow::all_dims(), `2::2`]
  y <- keras::k_stack(list(-x2, x1), axis = -1L)
  keras::k_reshape(y, keras::k_shape(x))
}

repeat_twice <- function(x) {
  keras::k_repeat_elements(x, 2L, axis = -1L)
}

#' ROtational Position Embeddings
#' @param x inputs
#' @param rots precomputed rotational matrix from \code{rotation_matrix}
#' @export
rope <- function(x, rots) {
  # based on equation 34 from Roformers paper

  # x = {batch_size, seqlen, num_heads, head_size}
  # freqs = {1, max_seqlen, 1, head_size}
  seqlen <- keras::k_shape(x)[2]

  # {2, 1, seqlen, 1, head_size}
  #rots <- rots[,,1:seqlen,,]
  c(rot_cos, rot_sin) %<-% keras::k_unstack(rots, axis = 1L)

  # {batch_size, seqlen, num_heads, head_size}
  y <- rotate_every_two(x)
  (x * rot_cos) + (y * rot_sin)
}
