test_that("rope_angles are correct", {
  feature_dim <- 16L
  theta <- 10000
  feature_dim_half <- feature_dim %/% 2

  # compute angles
  actual <- rope_angles(feature_dim, theta = theta)

  # from Roformers paper
  expected <- theta^(-(seq(feature_dim_half) - 1)/feature_dim_half)

  # check shape
  expect_equal(dim(actual), feature_dim_half)

  # check values
  expect_equal(as.vector(actual), expected, tolerance = 1E-7)
})

test_that("rope_matrix produces correct shape", {
  seqlen <- 8L
  feature_dim <- 18L
  theta <- 10000
  feature_dim_half <- feature_dim %/% 2

  rots <- rope_matrix(seqlen, feature_dim, theta = theta)

  # check shape
  c(rots_cos, rots_sin) %<-% keras::k_unstack(rots, axis = 1L)
  expect_equal(dim(rots), c(2, 1, seqlen, 1, feature_dim))
  expect_equal(dim(rots_cos), c(1, seqlen, 1, feature_dim))
  expect_equal(dim(rots_sin), c(1, seqlen, 1, feature_dim))
})

test_that("first token does not get rotated", {
  seqlen <- 8L
  feature_dim <- 18L
  theta <- 10000
  feature_dim_half <- feature_dim %/% 2

  rots <- rope_matrix(seqlen, feature_dim, theta = theta)
  c(rots_cos, rots_sin) %<-% keras::k_unstack(rots, axis = 1L)

  # check shape
  expect_equal(as.vector(rots_cos[,1,,]), rep(1, feature_dim))
  expect_equal(as.vector(rots_sin[,1,,]), rep(0, feature_dim))
})

test_that("rotate_every_two works", {
  x <- tf$constant(1:8)
  y <- rotate_every_two(x)
  expect_equal(as.vector(y), c(-2, 1, -4, 3, -6, 5, -8, 7))
})

test_that("rope works", {
  c(batch_size, seqlen, num_heads, head_size) %<-% c(7, 10, 3, 6)

  x <- tf$random$uniform(shape = shape(batch_size, seqlen, num_heads, head_size))
  rots <- rope_matrix(seqlen, head_size)

  rx <- rope(x, rots)

  # check shape
  expect_equal(dim(rx), dim(x))

  # check that first token is not rotated
  expect_equal(as.vector(rx[,1,,]), as.vector(x[,1,,]))

  # dot product of vector with itself is the same
  dot <- k_sum(x*x, axis = -1)
  rdot <- k_sum(rx*rx, axis = -1)
  expect_equal(as.vector(dot), as.vector(rdot), tolerance = 1E-7)
})
