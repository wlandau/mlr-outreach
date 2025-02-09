# Load packages and function -----------------------------------------------------------
source("packages.R")

sourceDirectory("R")

# Set Slurm options for workers -------------------------------------------

#options(clustermq.scheduler = "slurm",
#        clustermq.template = "slurm_clustermq.tmpl")
options(clustermq.scheduler = "multicore")

# Create plans ------------------------------------------------------------

benchmark_plan = code_to_plan("benchmark.R")

# Set the config ----------------------------------------------------------

drake_config(benchmark_plan, verbose = 2,# parallelism = "clustermq", #jobs = 2,
             #template = list(n_cpus = 5, memory = 34000, log_file = "worker%a.log"),
             #prework = quote(future::plan(future.callr::callr, workers = 4)),
             console_log_file = "drake.log", lock_envir = F)


