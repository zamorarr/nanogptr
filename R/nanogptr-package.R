#' @keywords internal
"_PACKAGE"

tf <- NULL
keras <- NULL
.onLoad <- function(libname, package) {
  tf <<- tensorflow::tf
  keras <<- keras::keras
}

## usethis namespace: start
#' @importFrom keras new_layer_class
#' @importFrom zeallot %<-%
## usethis namespace: end
NULL
