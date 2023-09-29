library(readxl)
library(tidyverse)
library(magrittr)
library(pbapply)
library(pracma)
library(text2vec)

source("src/R/functions/one_hot_encode.R")
source("src/R/functions/standardize.R")
source("src/R/functions/cross_platform_getcores.R")

tokens_to_sequence = function(x, words) {
    sapply(x, function(y) {
        if (is.element(y, words)) {
            return(which(words==y))
        } else {
            return(NaN)
        }
    }, USE.NAMES = F)
}

n.cores = cross_platform_getcores()
pboptions(type="timer")

retrospective_data = readRDS("data/multisite_data.rds")
code_data = readRDS("data/multisite_icd_cpt.rds")

survey_results = read_excel("data/pain_prediction/cleaned_survey_results.xlsx")
processed_data = read_excel("data/pain_prediction/pain_prediction_processed_data.xlsx")
processed_data$PreopPain %<>% replace_na(-1)
icd_codes = read_excel("data/pain_prediction/token_icds.xlsx")
cpt_codes = read_excel("data/pain_prediction/token_cpts.xlsx")

demographics = data.frame(
    age = processed_data$Age,
    weight = processed_data$WeightKG,
    height = processed_data$HeightCM * 0.393701, # Convert to inches
    gender = (processed_data$SexDSC == "Male") * 1,
    race = one.hot.encode(factor(processed_data$Race, levels=c("Asian", "Black", "Hispanic", "Other", "White"))),
    preop_pain = processed_data$PreopPain/10,
    surgery_service = one.hot.encode(factor(processed_data$ServiceDSC, levels=retrospective_data$service_levels)),
    surgery_urgency=one.hot.encode(factor(processed_data$CaseClassDSC, levels=retrospective_data$urgency_levels)),
    inpatient_vs_ambulatory=1
)

demographics$age %<>% standardize(mean_value=retrospective_data$means[1], sd_value=retrospective_data$sds[1])
demographics$weight %<>% standardize(mean_value=retrospective_data$means[2], sd_value=retrospective_data$sds[2])
demographics$height %<>% standardize(mean_value=retrospective_data$means[3], sd_value=retrospective_data$sds[3])

demographics %<>% as.matrix()
demographics[is.na(demographics)] = 0

### Process ICD + CPT codes
log_ids = processed_data$LogID

fprintf("Tokenizing ICD codes...\n")
processed_icd_codes = pblapply(log_ids, function(x) {
    current_icd = icd_codes %>% dplyr::filter(LogID == x)
    tokens = unique(current_icd$ICD)
    return(stringr::str_c(rep("icd_", length(tokens)), tokens))
}, cl=n.cores)

fprintf("Tokenizing CPT codes...\n")
processed_cpt_codes = pblapply(log_ids, function(x) {
    current_cpt_data = cpt_codes %>% dplyr::filter(LogID == x)
    tokens = unique(current_cpt_data$CPT)
    return(stringr::str_c(rep("cpt_", length(tokens)), tokens))
}, cl=n.cores)

fprintf("Combining tokens...\n")
combined_tokens = pbmapply(function(a,b) c(a,b), processed_icd_codes, processed_cpt_codes)
iterator = itoken(combined_tokens, ids=log_ids)

words = code_data$pruned_vocab$term
vectorizer = vocab_vectorizer(code_data$pruned_vocab)
dtm = create_dtm(iterator, vectorizer)

fprintf("Generating sequences...\n")
sequences = pblapply(combined_tokens, function(x) tokens_to_sequence(x, words), cl=n.cores)
icd_sequences = pblapply(processed_icd_codes, function(x) tokens_to_sequence(x, words), cl=n.cores)
cpt_sequences = pblapply(processed_cpt_codes, function(x) tokens_to_sequence(x, words), cl=n.cores)


pad_len = dim(code_data$sequences)[2]

fprintf("Padding sequences...\n")
padded_sequences = pblapply(sequences, function(x) {
    result = rep(0, pad_len)
    if (any(!is.na(x))) {
        result[seq_len(sum(!is.na(x)))] = x[!is.na(x)]
    }
    return(result)
}, cl=n.cores)
padded_sequences_all = do.call(rbind, padded_sequences)

padded_icd_sequences = pblapply(icd_sequences, function(x) {
    result = rep(0, pad_len)
    if (any(!is.na(x))) {
        result[seq_len(sum(!is.na(x)))] = x[!is.na(x)]
    }
    return(result)
}, cl=n.cores)
padded_icd_sequences_all = do.call(rbind, padded_icd_sequences)

padded_cpt_sequences = pblapply(cpt_sequences, function(x) {
    result = rep(0, pad_len)
    if (any(!is.na(x))) {
        result[seq_len(sum(!is.na(x)))] = x[!is.na(x)]
    }
    return(result)
}, cl=n.cores)
padded_cpt_sequences_all = do.call(rbind, padded_cpt_sequences)

saveRDS(
    list(
        pruned_vocab = code_data$pruned_vocab,
        sequences = padded_sequences_all,
        demographics = demographics,
        icd_sequences = padded_icd_sequences_all,
        cpt_sequences = padded_cpt_sequences_all,
        dtm = dtm,
        log_ids = log_ids
    ), file="data/pain_prediction/prospective_data_v2.rds"
)