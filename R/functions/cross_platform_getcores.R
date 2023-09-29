cross_platform_getcores = function() {
    # Parallelize according to functionality available to OS
    if (.Platform$OS.type == "unix") {
        cores = parallel::detectCores()
    } else {
        cores = 1
    }
    return(cores)
}
