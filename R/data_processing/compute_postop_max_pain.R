rm(list=ls())
library(tidyverse)
library(magrittr)
library(readxl)
library(lubridate)
library(parallel)
library(tictoc)

if (.Platform$OS.type == "unix") {
    cores = parallel::detectCores()/2
} else {
    cores = 1
}

all_data = readRDS("data/data14.rds")
pain_table = read.csv("data/converted/paintable_v2.csv")
pain_table$timestamp %<>% as_datetime()
timestamps = read_excel("data/converted/all_timestamps.xlsx")
timestamps$surgicalDurationStart %<>% as_datetime()

merged_pain_table = dplyr::left_join(pain_table, timestamps %>% dplyr::select(logID, surgicalDurationStart), by=c("log_id" = "logID"))
merged_pain_table %<>% na.omit()

tic("Computing max pain values")
max_pain_values = mcmapply(function(x) {
    current_pain = merged_pain_table %>% dplyr::filter(log_id == x)
    if (dim(current_pain)[1] == 0) {
        return(rep(NA, 5))
    }
    
    sapply(seq_len(5), function(y) {
        filtered_current_pain = current_pain %>% dplyr::filter(lubridate::date(timestamp) == lubridate::date(surgicalDurationStart) + lubridate::ddays(y-1))
        if (dim(filtered_current_pain)[1] == 0) {
            return(NA)
        } else {
            return(max(filtered_current_pain$value))
        }
    })
    
}, all_data$log_id, mc.cores=cores)
toc()

saveRDS(max_pain_values, file="data/postop_max_pain_5day.rds")