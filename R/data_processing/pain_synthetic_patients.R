library(tidyverse)
library(magrittr)
library(readxl)
library(comorbidity)

source("src/R/functions/standardize.R")
source("src/R/functions/one_hot_encode.R")

all_data = readRDS("prelude_data/all_data.rds")
multisite_data = read_excel("data/pain_prediction/other_site/pain_prediction_processed_data.xlsx")

service_levels = sort(unique(c(unique(all_data$surgery_service), unique(multisite_data$ServiceDSC))))
urgency_levels = sort(unique(c(unique(all_data$surgery_urgency), unique(multisite_data$CaseClassDSC))))

multisite_data = readRDS("data/multisite_data.rds")
code_data = readRDS("data/multisite_icd_cpt.rds")

patient_1 = data.frame(
    age = 41,
    gender = "Female",
    weight = 65,
    height = 1.593,
    preop_pain = 5,
    race = "White",
    surgery_service = "Orthopedic Surgery",
    surgery_urgency = "Non-urgent",
    inpatient_vs_ambulatory = "Inpatient"
)

patient_1_codes = c("cpt_22552", "icd_G89.4", "icd_Z72.0", "icd_G47.9", "icd_M43.20")

patient_2 = data.frame(
    age = 74,
    gender = "Male",
    weight = 74,
    height = 1.68,
    preop_pain = 0,
    race = "Black",
    surgery_service = "Urology",
    surgery_urgency="Non-urgent",
    inpatient_vs_ambulatory = "Outpatient"
)

patient_2_codes = c("cpt_52260", "icd_Z72.0", "icd_F11.29", "icd_E78.00", "icd_I10")

patient_3 = data.frame(
    age = 58,
    gender = "Female",
    weight = 79,
    height = 1.55,
    preop_pain = 2,
    race = "Asian",
    surgery_service = "Otolaryngology",
    surgery_urgency = "Non-urgent",
    inpatient_vs_ambulatory = "Inpatient"
)

patient_3_codes = c("cpt_31255", "icd_M79.7", "icd_F32.A", "icd_J34.1", "icd_E03.9")


# Generate demographics
all_patients = rbind(patient_1, patient_2, patient_3)


demographics = data.frame(
    age=all_patients$age,
    weight=all_patients$weight,
    height=all_patients$height * 39.3701, 
    gender=(all_patients$gender=="Male")*1, 
    race=one.hot.encode(factor(all_patients$race, levels=c("Asian", "Black", "Hispanic", "Other", "White"))),
    preop_pain = all_patients$preop_pain/10,
    surgery_service=one.hot.encode(factor(all_patients$surgery_service, levels=service_levels)),
    surgery_urgency=one.hot.encode(factor(all_patients$surgery_urgency, levels=urgency_levels)),
    inpatient_vs_ambulatory=(all_patients$inpatient_vs_ambulatory=="Inpatient")*1
)

demographics$age %<>% standardize(mean_value=multisite_data$means[1], sd_value=multisite_data$sds[1])
demographics$weight %<>% standardize(mean_value=multisite_data$means[2], sd_value=multisite_data$sds[2])
demographics$height %<>% standardize(mean_value=multisite_data$means[3], sd_value=multisite_data$sds[3])

demographics %<>% as.matrix()
demographics[is.na(demographics)] = 0

# Generate and pad sequences
codes = list(patient_1_codes, patient_2_codes, patient_3_codes)

icd_codes = data.frame(
    id = lapply(seq_along(codes), function(x) rep(x, length(codes[[x]])-1)) %>% do.call(c, .),
    code = lapply(codes, function(x) lapply(str_split(x[-1], "_"), function(y) y[2]) %>% do.call(c, .)) %>% do.call(c, .)
)
comorbidities = comorbidity(icd_codes, id="id", code="code", assign0=F, map="elixhauser_icd10_quan")
elixhauser_score = score(comorbidities, weights="vw", assign0=T)

comorbidities = comorbidity(icd_codes, id="id", code="code", assign0=F, map="charlson_icd10_quan")
charlson_score = score(comorbidities, weights="charlson", assign0=T)

words = code_data$pruned_vocab$term
sequences = lapply(codes, function(x) {
    sapply(x, function(y) {
        if (is.element(y, words)) {
            return(which(words==y))
        } else {
            return(NaN)
        }
    }, USE.NAMES = F)
})
pad_len = dim(code_data$sequences)[2]

padded_sequences = lapply(sequences, function(x) {
    result = rep(0, pad_len)
    if (any(!is.na(x))) {
        result[seq_len(sum(!is.na(x)))] = x[!is.na(x)]
    }
    return(result)
})

padded_sequences_all = do.call(rbind, padded_sequences)

saveRDS(list(demographics=demographics, sequences=padded_sequences_all, elixhauser=elixhauser_score, charlson=charlson_score, patients=all_patients, codes=codes), file="data/pain_synthetic_patients.rds")