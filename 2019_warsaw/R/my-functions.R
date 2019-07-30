#' @title mlr3::benchmark() wrapper
#'

benchmark_custom <- function(design, ctrl = NULL, cpus = NULL) {

  plan("multicore", workers = cpus)
  set.seed(12345)

  bmr <- benchmark(design = design, ctrl = ctrl)

  return(bmr)
}
