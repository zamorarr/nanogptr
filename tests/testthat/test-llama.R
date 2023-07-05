test_that("hidden dim is calculated correctly", {
  # Rounds 8/3*output_dim to the next multiple of multiple_of
  expect_equal(llama_hidden_dim(4, 8), 16)
  expect_equal(llama_hidden_dim(3, 8), 8)
  expect_equal(llama_hidden_dim(12, 2), 32)
  expect_equal(llama_hidden_dim(64, 8), 176)
})
