map_cpt_icd_descriptions = function(codes, icd_descriptions, cpt_descriptions) {
    split_codes = str_split(codes, "_")
    all_codes = sapply(split_codes, function(x) x[2])
    all_descriptions = rep(NA, length(all_codes))
    is_icd = sapply(split_codes, function(x) x[1] == "icd")
    is_cpt = sapply(split_codes, function(x) x[1] == "cpt")
    mapped_icd_descriptions = dplyr::left_join(data.frame(ICD=all_codes[is_icd]), icd_descriptions, by="ICD")
    all_descriptions[is_icd] = mapped_icd_descriptions$Description
    mapped_cpt_descriptions = dplyr::left_join(data.frame(CPTCD=all_codes[is_cpt]), cpt_descriptions, by="CPTCD")
    all_descriptions[is_cpt] = mapped_cpt_descriptions$CPTDSC
    return(all_descriptions)
}