# 7B llama with only one layer
Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")

# pytorch
torch <- reticulate::import("torch", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)
transformers <- reticulate::import("transformers", convert = FALSE)

model3 <- transformers$AutoModelForCausalLM$from_pretrained(
  normalizePath("~/data/llama-hf/bobby"),
  torch_dtype = torch$float16
)
model3$to("cuda")

# tensorflow
params <- list(
  dim = 4096L,
  multiple_of = 256L,
  n_heads = 32L,
  n_layers = 1L,
  norm_eps = 1E-6,
  vocab_size = 32000L#as.vector(tokenizer$vocab_size())
)

# create model
library(keras)
library(tensorflow)
devtools::load_all()

keras::k_set_floatx("float16")
model <- llama_model(params)
llama_load_weights(model, params, path = "~/data/llama-np/7B") # original weights

# compare pytorch output to keras outputs
x <- array(seq.int(4), dim = c(1,4))

inspect <- reticulate::import("inspect", convert = TRUE)
cat(inspect$getsource(model3$model$layers[0]$forward))
cat(inspect$getsource(model3$model$layers[0]$`__init__`))
cat(inspect$getsource(model3$model$layers[0]$self_attn$forward))
cat(inspect$getsource(model3$model$norm$forward))
cat(inspect$getsource(model3$model$layers$`__call__`))
cat(inspect$getsource(model3$model$forward))
cat(inspect$getsource(model3$model$layers[0]$self_attn$rotary_emb$forward))
cat(inspect$getsource(model3$model$layers[0]$self_attn$forward))
cat(inspect$getsource(model3$model$layers[0]$mlp$forward))
cat(inspect$getsource(model3$model$layers[0]$mlp$gate_proj$forward))

# with(torch$no_grad(), {
#   torch$tensor(x)$to("cuda") |>
#     model3$model$embed_tokens() |>
#     model3$model$layers[[0]]$forward() |> # this is not going to work because I don't have attention mask or position id args
#     reticulate::py_get_item(0L) |>
#     model3$model$norm() |>
#     model3$lm_head()
# })

pt_output <- with(torch$no_grad(), {
  torch$tensor(x)$to("cuda") |>
    model3$model$forward() |>
    reticulate::py_get_item("last_hidden_state") #|>
    #model3$lm_head()
})

pt_output <- with(torch$no_grad(), {
  torch$tensor(x)$to("cuda") |>
    model3$forward() |>
    reticulate::py_get_item("logits")
})

tf_output <- tf$constant(x) |>
  model$layers[[2]]$call() |> # embedding
  model$layers[[3]]$call() |> # transformer
  model$layers[[4]]$call() #|> # norm
  #model$layers[[5]]$call()    # output

tf_output <- tf$constant(x) |> model()

reticulate::py_to_r(np$array_equal(pt_output$cpu()$numpy(), tf_output$numpy()))
#reticulate::py_to_r(np$array_equal(pt_output[0]$cpu()$numpy(), tf_output[1]$numpy()))
mean(reticulate::py_to_r(np$isclose(tf_output$numpy(), pt_output$cpu()$numpy(), atol=1E-5)))

# test mlp
input <- array(seq(3*4096)/(3*4096), dim = c(1, 3, 4096))
with(torch$no_grad(), {
  torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$mlp()
})

tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$ffwd()

# test self-attention
with(torch$no_grad(), {
  torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$self_attn()
})

rots <- rope_matrix(3, params$dim %/% params$n_heads)
tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$sa(rots)

# test w1
with(torch$no_grad(), {
  torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$mlp$gate_proj() |>
    model3$model$layers[0]$mlp$act_fn()
})

tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$ffwd$w1() |>
  tf$nn$silu()

# test w3
with(torch$no_grad(), {
  torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$mlp$up_proj()
})

tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$ffwd$w3()

# test w1*w3
with(torch$no_grad(), {
  r1 <- torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$mlp$gate_proj() |>
    model3$model$layers[0]$mlp$act_fn()
  r3 <- torch$tensor(input, dtype = torch$float16)$to("cuda") |>
    model3$model$layers[0]$mlp$up_proj()

  model3$model$layers[0]$mlp$down_proj(torch$mul(r1, r3))
})

s1 <- tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$ffwd$w1() |>
  tf$nn$silu()

s3 <- tf$constant(input, dtype = tf$dtypes$float16) |>
  model$layers[[3]]$ffwd$w3()

model$layers[[3]]$ffwd$w2(s1 * s3)

mean(reticulate::py_to_r(r1$detach()$cpu()$numpy()) == s1$numpy())
reticulate::py_to_r(torch$mul(r1, r3)[0][0]$detach()$cpu()$numpy()) == tf$multiply(s1, s3)[1,1,]$numpy()
     #model3$model$layers[0]$mlp$gate_proj$weight#$detach()$cpu()$numpy()
#model$layers[[3]]$ffwd$w1$weights[[1]]#$numpy()
