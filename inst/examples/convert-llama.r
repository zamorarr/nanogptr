library(dplyr)
library(stringr)
library(glue)

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
Sys.setenv("XLA_FLAGS" = "--xla_gpu_cuda_data_dir=/usr/lib/cuda")
reticulate::use_condaenv("r-reticulate")

torch <- reticulate::import("torch", convert = FALSE)
np <- reticulate::import("numpy", convert = FALSE)

# params
model_path <- normalizePath("~/data/llama-hf/llama-7b-hf")
outdir <- "~/data/llama-np/7B"

# file map
j <- jsonlite::read_json(file.path(model_path, "pytorch_model.bin.index.json"))

weight_map <- tibble::tibble(hf_file = unname(unlist(j$weight_map)), hf_name = names(j$weight_map)) |>
  mutate(
    block_id = str_match(hf_name, "model\\.layers\\.([0-9]+)")[,2],
    hf_file = file.path(model_path, hf_file),
    llama_name = case_when(
      str_detect(hf_name, "self_attn.q_proj") ~ glue("layers.{block_id}.attention.wq.weight"),
      str_detect(hf_name, "self_attn.k_proj") ~ glue("layers.{block_id}.attention.wk.weight"),
      str_detect(hf_name, "self_attn.v_proj") ~ glue("layers.{block_id}.attention.wv.weight"),
      str_detect(hf_name, "self_attn.o_proj") ~ glue("layers.{block_id}.attention.wo.weight"),
      str_detect(hf_name, "mlp.gate_proj") ~ glue("layers.{block_id}.feed_forward.w1.weight"),
      str_detect(hf_name, "mlp.down_proj") ~ glue("layers.{block_id}.feed_forward.w2.weight"),
      str_detect(hf_name, "mlp.up_proj") ~ glue("layers.{block_id}.feed_forward.w3.weight"),
      str_detect(hf_name, "input_layernorm") ~ glue("layers.{block_id}.attention_norm.weight"),
      str_detect(hf_name, "post_attention_layernorm") ~ glue("layers.{block_id}.ffn_norm.weight"),
      hf_name == "model.embed_tokens.weight" ~ "tok_embeddings.weight",
      hf_name == "model.norm.weight" ~ "norm.weight",
      hf_name == "lm_head.weight" ~ "output.weight",
      TRUE ~ NA_character_
    ),
    keras_name = case_when(
      str_detect(hf_name, "self_attn.q_proj") ~ glue("transformer_{block_id}/attention/wq/kernel:0"),
      str_detect(hf_name, "self_attn.k_proj") ~ glue("transformer_{block_id}/attention/wk/kernel:0"),
      str_detect(hf_name, "self_attn.v_proj") ~ glue("transformer_{block_id}/attention/wv/kernel:0"),
      str_detect(hf_name, "self_attn.o_proj") ~ glue("transformer_{block_id}/attention/wo/kernel:0"),
      str_detect(hf_name, "mlp.gate_proj") ~ glue("transformer_{block_id}/feed_forward/w1/kernel:0"),
      str_detect(hf_name, "mlp.down_proj") ~ glue("transformer_{block_id}/feed_forward/w2/kernel:0"),
      str_detect(hf_name, "mlp.up_proj") ~ glue("transformer_{block_id}/feed_forward/w3/kernel:0"),
      str_detect(hf_name, "input_layernorm") ~ glue("transformer_{block_id}/attention_norm/Variable:0"),
      str_detect(hf_name, "post_attention_layernorm") ~ glue("transformer_{block_id}/ffn_norm/Variable:0"),
      hf_name == "model.embed_tokens.weight" ~ "tok_embeddings/embeddings:0",
      hf_name == "model.norm.weight" ~ "norm/Variable:0",
      hf_name == "lm_head.weight" ~ "output/kernel:0",
      TRUE ~ NA_character_
    )
) |>
  filter(!is.na(llama_name)) |>
  select(-block_id)

readr::write_csv(weight_map[-1], "inst/examples/llama-7b-layers.csv")

weight_split <- split(weight_map, weight_map$hf_file)

write_file <- function(filename, outdir, hf_names, llama_names) {
  filename <- normalizePath(filename, mustWork = TRUE)
  outdir <- normalizePath(outdir, mustWork = TRUE)

  weights <- torch$load(filename, map_location = "cpu")

  mapply(function(hf, llama) {
    w <- weights[[hf]]$numpy()
    outfile <- file.path(outdir, paste0(llama, ".npy"))
    np$save(outfile, w)
    message(glue("wrote '{outfile}' with shape {w$shape}"))
  }, hf_names, llama_names)

  invisible(outdir)
}

write_weights <- function(data, outdir) {
  for (i in seq_along(data)) {
    write_file(data[[i]]$hf_file[1], outdir, data[[i]]$hf_name, data[[i]]$llama_name)
  }
}

write_weights(weight_split, outdir)
