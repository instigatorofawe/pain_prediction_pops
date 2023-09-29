library(tidyverse)
library(magrittr)
library(pbapply)
library(readxl)
library(pracma)
library(text2vec)
pboptions(type="timer")

source("src/R/functions/standardize.R")
source("src/R/functions/one_hot_encode.R")
source("src/R/functions/cross_platform_getcores.R")

n.cores = cross_platform_getcores()

read_icd_cpt = function() {
    n.cores = cross_platform_getcores()
    all_data = readRDS("data/all_data.rds")
    icd_codes_1 = read_excel("data/icds_v2_1.xlsx", col_types=c("numeric", "text", "date", "date", "date", "text"))
    icd_codes_2 = read_excel("data/icds_v2_2.xlsx", col_types=c("numeric", "text", "date", "date", "date", "text"))
    icd_codes_3 = read_excel("data/icds_v2_3.xlsx", col_types=c("numeric", "text", "date", "date", "date", "text"))
    
    primary_cpt_data = read_excel("data/CPT_2022_08_02.xlsx")
    
    icd_codes = rbind(icd_codes_1, icd_codes_2, icd_codes_3) %>% dplyr::filter(CurrentICD10ListTXT != "NULL" & is.element(LogID, all_data$log_id) & DiagnosisDTS < SurgeryDTS)
    fprintf("Tokenizing ICD codes...\n")
    processed_icd_codes = pblapply(all_data$log_id, function(x) {
        current_icd = icd_codes %>% dplyr::filter(LogID == x)
        tokens = unique(str_trim(do.call(c, sapply(current_icd$CurrentICD10ListTXT, function(y) str_split(y, ","), USE.NAMES = F))))
        return(stringr::str_c(rep("icd_", length(tokens)), tokens))
    }, cl=n.cores)
    
    fprintf("Tokenizing CPT codes...\n")
    processed_cpt_codes = pblapply(all_data$log_id, function(x) {
        current_cpt_data = primary_cpt_data %>% dplyr::filter(LogID == x)
        tokens = unique(current_cpt_data$CPTCD)
        return(stringr::str_c(rep("cpt_", length(tokens)), tokens))
    }, cl=n.cores)
    
    fprintf("Combining tokens...\n")
    combined_tokens = mapply(function(a,b) c(a,b), processed_icd_codes, processed_cpt_codes)
    return(list(icd=processed_icd_codes, cpt=processed_cpt_codes, combined=combined_tokens))
}


read_new_icd_cpt = function() {
    n.cores = cross_platform_getcores()
    demographics = read_excel("data/pain_prediction/other_site/pain_prediction_processed_data.xlsx")
    log_ids = demographics$LogID

    icd1 = read_excel("data/pain_prediction/other_site/token_icds1.xlsx")
    icd2 = read_excel("data/pain_prediction/other_site/token_icds2.xlsx")
    icd_codes = rbind(icd1, icd2)
    
    cpt = read_excel("data/pain_prediction/other_site/token_cpts.xlsx", col_types = c("numeric", "text"))
    
    fprintf("Tokenizing ICD codes...\n")
    processed_icd_codes = pblapply(log_ids, function(x) {
        current_icd = icd_codes %>% dplyr::filter(LogID == x)
        tokens = unique(current_icd$ICD)
        return(stringr::str_c(rep("icd_", length(tokens)), tokens))
    }, cl=n.cores)
    
    fprintf("Tokenizing CPT codes...\n")
    processed_cpt_codes = pblapply(log_ids, function(x) {
        current_cpt_data = cpt %>% dplyr::filter(LogID == x)
        tokens = unique(current_cpt_data$CPT)
        return(stringr::str_c(rep("cpt_", length(tokens)), tokens))
    }, cl=n.cores)
    
    fprintf("Combining tokens...\n")
    combined_tokens = pbmapply(function(a,b) c(a,b), processed_icd_codes, processed_cpt_codes)
    
    return(list(icd=processed_icd_codes, cpt=processed_cpt_codes, combined=combined_tokens))
}

# MGH data
all_data = readRDS("data/all_data.rds")
multisite_data = read_excel("data/pain_prediction/other_site/pain_prediction_processed_data.xlsx")
multisite_data$PreopPain %<>% replace_na(-1)

service_levels = sort(unique(c(unique(all_data$surgery_service), unique(multisite_data$ServiceDSC))))
urgency_levels = sort(unique(c(unique(all_data$surgery_urgency), unique(multisite_data$CaseClassDSC))))

demographics = data.frame(
    age=all_data$age,
    weight=all_data$weight,
    height=all_data$height, 
    gender=(all_data$gender=="Male")*1, 
    race=one.hot.encode(factor(all_data$race, levels=c("Asian", "Black", "Hispanic", "Other", "White"))),
    preop_pain = all_data$preop_pain_score/10,
    surgery_service=one.hot.encode(factor(all_data$surgery_service, levels=service_levels)),
    surgery_urgency=one.hot.encode(factor(all_data$surgery_urgency, levels=urgency_levels)),
    inpatient_vs_ambulatory=all_data$inpatient_vs_ambulatory
)

# Multisite data
multisite_demographics = data.frame(
    age=multisite_data$Age,
    weight=multisite_data$WeightKG,
    height=multisite_data$HeightCM/2.54, # Convert to inches
    gender=(multisite_data$SexDSC=="Male")*1,
    race=one.hot.encode(factor(multisite_data$Race, levels=c("Asian", "Black", "Hispanic", "Other", "White"))),
    preop_pain = multisite_data$PreopPain/10,
    surgery_service=one.hot.encode(factor(multisite_data$ServiceDSC, levels=service_levels)),
    surgery_urgency=one.hot.encode(factor(multisite_data$CaseClassDSC, levels=urgency_levels)),
    inpatient_vs_ambulatory=(multisite_data$InptVAmb == "Inpatient")*1
)

# Outcomes
postop_max_pain = readRDS("data/postop_max_pain_5day.rds")
multisite_max_pain = multisite_data %>% select(PostopPainDay0, PostopPainDay1, PostopPainDay2, PostopPainDay3, PostopPainDay4)

# Combine
mgh_location = rep("MGH", dim(all_data)[1])
all_locations = c(mgh_location, multisite_data$LocationAbbrev)

all_log_ids = c(all_data$log_id, multisite_data$LogID)
all_demographics = rbind(demographics, multisite_demographics)

all_outcomes = rbind(t(postop_max_pain), multisite_max_pain %>% as.matrix)
all_outcomes[is.na(all_outcomes)] = NaN

means = c(mean(all_demographics$age, na.rm=T), mean(all_demographics$weight, na.rm=T), mean(all_demographics$height, na.rm=T))
sds = c(sd(all_demographics$age, na.rm=T), sd(all_demographics$weight, na.rm=T), sd(all_demographics$height, na.rm=T))

all_demographics$age %<>% standardize()
all_demographics$weight %<>% standardize()
all_demographics$height %<>% standardize()

all_demographics %<>% as.matrix()
all_demographics[is.na(all_demographics)] = 0

set.seed(0)
training_set = readRDS("data/pain_prediction_combined_training_set.rds")
new_training_set = readRDS("data/multisite_training_set.rds")

training = c(training_set$training, new_training_set$training)
validation = c(!training_set$validation, new_training_set$validation)

saveRDS(
    list(
        log_id = all_log_ids,
        location = all_locations,
        demographics = all_demographics,
        outcomes = all_outcomes,
        training = training,
        validation = validation,
        service_levels = service_levels,
        urgency_levels = urgency_levels,
        means = means,
        sds = sds
    ),
    file="data/multisite_data.rds"
)

# MGH
mgh_data = read_icd_cpt()
processed_icd_codes = mgh_data$icd
processed_cpt_codes = mgh_data$cpt
combined_tokens = mgh_data$combined

# Multi-site data
new_data = read_new_icd_cpt()
new_icd_codes = new_data$icd
new_cpt_codes = new_data$cpt
new_tokens = new_data$combined

all_tokens = c(combined_tokens, new_tokens)
all_icd = c(processed_icd_codes, new_icd_codes)
all_cpt = c(processed_cpt_codes, new_cpt_codes)

# Generate sequences
iterator = itoken(all_tokens, ids=all_data$log_id)
vocab = create_vocabulary(iterator)
pruned_vocab = prune_vocabulary(vocab,  term_count_min = 2,  doc_proportion_max = 1, doc_proportion_min = 1e-4)

words = pruned_vocab$term
vectorizer = vocab_vectorizer(pruned_vocab)
dtm = create_dtm(iterator, vectorizer)

tokens_to_sequence = function(x) {
    sapply(x, function(y) {
        if (is.element(y, words)) {
            return(which(words==y))
        } else {
            return(NaN)
        }
    }, USE.NAMES = F)
}

sequences = pblapply(all_tokens, tokens_to_sequence, cl=n.cores)
icd_sequences = pblapply(all_icd, tokens_to_sequence, cl=n.cores)
cpt_sequences = pblapply(all_cpt, tokens_to_sequence, cl=n.cores)

doc_len = sapply(sequences, function(x) sum(!is.na(x)))
pad_len = max(doc_len)

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
        sequences = padded_sequences_all,
        icd_sequences = padded_icd_sequences_all,
        cpt_sequences = padded_cpt_sequences_all,
        pruned_vocab = pruned_vocab,
        vocab = vocab
    ),
    file="data/multisite_icd_cpt.rds"
)