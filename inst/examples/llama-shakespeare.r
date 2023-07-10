Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")

library(tensorflow)
library(keras)
library(tfdatasets)
tf_config()
devtools::load_all()
#tf$test$is_gpu_available()

# training params
batch_size <- 32L
block_size <- 64L
learning_rate <- 3E-4
max_epochs <- 10L

# architecture params
params <- list(
  dim = 64L,
  multiple_of = 8L,
  n_heads = 4L,
  n_layers = 3L,
  norm_eps = 1E-6,
  vocab_size = -1L,
  max_seqlen = block_size
)

# get tiny shakespeare dataset
input_path <- "inst/examples/data/input.txt"
if (!file.exists(input_path)) {
  if (!dir.exists(dirname(input_path))) dir.create(dirname(input_path), recursive = TRUE)
  download.file(
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    input_path
  )
}

# read it in
text <- readr::read_file(input_path)

# unique characters in this text
chars <- sort(unique(strsplit(text, "")[[1]]))
vocab_size <- length(chars)
params$vocab_size <- vocab_size

# mapping from characters to integers
stoi <- setNames(seq_along(chars) - 1L, chars)
encode <- function(x) unname(stoi[strsplit(x, "")[[1]]]) # note not zero-indexed
decode <- function(x) paste0(chars[x + 1L], collapse = "")
# encode the entire dataset and store it in a tensor
data <- tf$constant(encode(text), dtype = tf$dtypes$int32)

# alternate mapping using keras text vectorization
#text_vectorizer <- layer_text_vectorization(standardize = NULL, split = "character")
#adapt(text_vectorizer, text)
#data2 <- text_vectorizer(text)
#setdiff(get_vocabulary(text_vectorizer, FALSE), chars)

# split into train/val sets
n <- as.integer(0.9*length(data))
train_data <- data[1:n]
val_data <- data[(n+1):length(data)]
stopifnot(length(train_data) + length(val_data) == length(data))

# build data pipeline
build_pipeline <- function(x, block_size) {
  tensor_slices_dataset(x) |>
    dataset_batch(block_size + 1, drop_remainder = TRUE) |>
    dataset_map(function(x) {
      input_text <- x[1:block_size]
      target_text <- x[2:(block_size + 1)]
      list(input_text, target_text)
    }) |>
    dataset_shuffle(1000) |>
    dataset_batch(batch_size, drop_remainder = TRUE) |>
    dataset_prefetch() |>
    dataset_cache()
}

dataset_train <- build_pipeline(train_data, block_size)
dataset_val <- build_pipeline(val_data, block_size)

# build model
model <- llama_model2(params)
#lapply(model$layers, \(layer) vapply(layer$weights, \(x) x$name, character(1)))
#model$count_params()
#plot(model, show_shapes = TRUE)

# callbacks
cosine_decay <- new_learning_rate_schedule_class(
  "CosineDecay",
  initialize = function(init_lr, decay_steps, alpha = 0) {
    # min_lr = init_lr*alpha
    self$init_lr <- init_lr
    self$decay_steps <- decay_steps
    self$alpha <- alpha
  },
  call = function(step) {
    step <- min(step, self$decay_steps)
    step_ratio <- step/self$decay_steps
    cosine_decay <- 0.5*(1 + cos(pi*step_ratio))
    decayed <- (1 - self$alpha)*cosine_decay + self$alpha
    self$init_lr*decayed
  },
  get_config = function() {
    # get_config() is abstract in the parent class
    # must override, not inherit
    # config <- super$get_config()
    config <- list()
    config$init_lr = self$init_lr
    config$decay_steps = self$decay_steps
    config$alpha = self$alpha
    config
  }
)

# compile
compile(
  model,
  loss = list(output = loss_sparse_categorical_crossentropy(from_logits = TRUE)),
  optimizer = optimizer_adam(cosine_decay(learning_rate, length(dataset_train)*max_epochs, alpha = 0.1))
)

evaluate(model, dataset_val)
c(x_sample, y_sample) %<-% next_batch(dataset_train)
model(x_sample)

cache <- model$create_cache(batch_size, max_seqlen = 100)
model(x_sample, cache = cache)

model$generate(as_tensor(array(c(16L, 42L), dim = c(1,2))))
model$generate <- tf_function(model$generate, autograph = TRUE, jit_compile = TRUE)

model <- llama_model2(params)
llama_generate(model, as_tensor(array(c(16L, 42L), dim = c(1,2))))
generate <- tf_function(llama_generate)
generate(model, as_tensor(array(16L, dim = c(1,1))))

# debugging above error
g <- reticulate::py_func(model$generate)
formals(model$generate)


# fit
history <- fit(
  model,
  dataset_train,
  validation_data = dataset_val,
  epochs = max_epochs
  #callbacks = list(callback_model_checkpoint("checkpoints/llama-shakespeare", save_best_only = TRUE))
)

# dummy input
dummy_prompt <- encode("BOBBY:\n")
dummy_input <- tf$constant(dummy_prompt, shape = shape(1, length(dummy_prompt)), dtype = tf$dtypes$int32)
dummy_output <- model$generate(dummy_input, 10L)
#dummy_output <- generate(model, dummy_input, 100)
cat(decode(dummy_output$numpy()[1,]))
