g2s<-function(...){
  inputs=list(...);
  return(g2sInterface(inputs));
}

g2s_schema_to_legacy<-function(result){
  if (!is.list(result) || is.null(names(result))) {
    return(result)
  }

  legacy <- list()
  if (!is.null(result$simulation)) {
    legacy[[length(legacy)+1]] <- result$simulation
  }

  artifact_names <- character()
  if (!is.null(result$artifacts) && !is.null(names(result$artifacts))) {
    artifact_names <- setdiff(names(result$artifacts), c("log", "warning", "error", "progress", "meta", "simulation"))
  }
  for (name in artifact_names) {
    if (!is.null(result[[name]])) {
      legacy[[length(legacy)+1]] <- result[[name]]
    }
  }

  if (!is.null(result$time)) {
    legacy[[length(legacy)+1]] <- result$time
  }

  meta <- result[setdiff(names(result), c("simulation", "time", "job_id", "status", "progress", "artifacts", "error", "warnings", artifact_names))]
  if (length(meta) > 0) {
    legacy[[length(legacy)+1]] <- meta
  }

  if (!is.null(result$progress)) {
    legacy[[length(legacy)+1]] <- result$progress
  }
  if (!is.null(result$job_id)) {
    legacy[[length(legacy)+1]] <- result$job_id
  }

  if (length(legacy) == 1) {
    return(legacy[[1]])
  }
  legacy
}
