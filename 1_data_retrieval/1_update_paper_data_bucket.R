library(arrow)
library(tidyverse)
library(readr)
library(viridis)


folder_path <- "data_update_2025/via bucket/articles/"

file_list <- list.files(folder_path, pattern = "\\.parquet$", full.names = TRUE)

updated_lattes <- file_list %>%
  lapply(read_parquet) %>%
  bind_rows()

names(updated_lattes)

######text prepro#####

article_table <- updated_lattes %>% 
  select(Titulo,Ano,DOI,ISSN) %>% 
  unique()

article_table <- article_table %>% 
  distinct(Titulo,.keep_all = TRUE)

text_table <- article_table %>% 
  mutate(id = Titulo) %>% 
  mutate(Titulo = gsub("[[:punct:]]", " ", Titulo)) %>% 
  mutate(Titulo = gsub("Author", "", Titulo)) %>% 
  mutate(Titulo = gsub("Reply", "", Titulo)) %>% 
  mutate(Titulo = gsub("Introduction", "", Titulo)) %>% 
  mutate(Titulo = gsub("Introdução", "", Titulo)) %>% 
  mutate(Titulo = gsub("Poster Sessions", "", Titulo)) %>% 
  mutate(Titulo = gsub("preparation", "", Titulo)) %>% 
  mutate(Titulo = str_squish(Titulo)) %>% 
  filter(Titulo != "") %>% 
  filter(!is.na(Titulo)) %>% 
  mutate(c = nchar(Titulo)) %>% 
  filter(c > 10) %>% 
  select(-c)

text_table <- text_table %>% 
  select(id,DOI,Titulo )


write_parquet(text_table,"data_update_2025/update_text_table.parquet")
text_table <- read_parquet("data_update_2025/update_text_table.parquet")

###meta#####

meta_lattes <- updated_lattes %>% 
  select(-IDLattes,-Nome) %>% unique() %>% 
  filter(Titulo %in% text_table$id)

meta_lattes$DOI[meta_lattes$DOI == ""] <- NA
meta_lattes$DOI[nchar(meta_lattes$DOI) <7] <- NA
meta_lattes$ISSN[nchar(meta_lattes$ISSN) <3] <- NA

meta_lattes <- meta_lattes %>% 
  distinct(Titulo, .keep_all = TRUE)

write_parquet(meta_lattes,"data_update_2025/update_meta_lattes.parquet")

meta_lattes <- read_parquet("data_update_2025/update_meta_lattes.parquet")
#max(meta_lattes$Ano)

###gender inference table##########

inference_table <- updated_lattes %>% 
  select(IDLattes,Nome) %>% 
  unique() %>% 
  filter(!is.na(Nome)) %>% 
  filter(Nome!="") %>% 
  rename("author_id" = "IDLattes") %>% 
  mutate(country = "Brazil") %>% 
  mutate(first_name = word(Nome),
         last_name =  word(Nome,2,-1)) %>%
  mutate(first_name = ifelse(first_name == "Ed", word(last_name), first_name),
         last_name = ifelse(first_name == "Ed", word(last_name,2,-1), last_name)) %>% 
  select(-Nome)

n_distinct(inference_table$author_id)

write_parquet(inference_table,"data_update_2025/update_inference_table_lattes.parquet")

rm(inference_table)

######fractional counting#######

author_gender <- read_parquet("data_update_2025/update_lattes_inference.parquet") %>% 
  mutate(gender = case_when(gender == "M" ~ "Men",
                            gender == "F" ~ "Women",
                            TRUE ~ gender))

1-n_distinct(author_gender$author_id[!author_gender$gender%in%c("Men","Women")])/n_distinct(author_gender$author_id)
#0.8823773

author_gender <- read_parquet("data_update_2025/update_lattes_inference.parquet") %>% 
  mutate(gender = case_when(gender == "M" ~ "Men",
                            gender == "F" ~ "Women",
                            TRUE ~ gender))%>% 
  filter(gender %in% c("Women","Men"))

frac_table <- updated_lattes %>%
  select(Titulo,IDLattes) %>% 
  unique() %>% 
  left_join(author_gender, by = c("IDLattes" = "author_id"))

# frac_table %>%
#   group_by(gender) %>%
#   summarise(n())

gender_fractional <- frac_table %>% 
  filter(!is.na(gender)) %>% 
  group_by(Titulo,gender) %>% 
  summarise(n=n_distinct(IDLattes)) %>% 
  group_by(Titulo) %>% 
  mutate(p = n/sum(n)) %>% 
  select(-n) %>% 
  pivot_wider(names_from = gender, values_from = p, values_fill = 0)

gender_fractional %>% 
  write_parquet("data_update_2025/update_gender_fractional.parquet")

#check
updated_lattes %>% 
  filter(!Titulo%in%gender_fractional$Titulo) %>% 
  pull(Nome)

######journal information quality#######

journal_info <- meta_lattes %>% 
  mutate(ISSN = ifelse(is.na(ISSN), "no ISSN", "has ISSN")) %>% 
  mutate(Veiculo = ifelse(is.na(Veiculo), "no journal name", "has journal name")) %>% 
  group_by(ISSN,Veiculo) %>% 
  summarise(n = n_distinct(id))

journal_info %>% 
  ggplot(aes(x = ISSN,y=Veiculo, fill = n))+
  geom_tile()+
  geom_text(aes(label = n))+
  theme_minimal()+
  scale_fill_viridis(begin = .3,end = .7)+
  theme(legend.position = "none")+
  labs(x = "", y="")

ggsave("results_update_2025/paper_draft/annex_check_journal.png", bg = "white", width = 4, height = 4) 

journal_info_year <- meta_lattes %>% 
  mutate(ISSN = ifelse(is.na(ISSN), "no ISSN", "has ISSN")) %>% 
  mutate(Veiculo = ifelse(is.na(Veiculo), "no journal name", "has journal name")) %>% 
  mutate(cat = paste0(ISSN, " & ",Veiculo)) %>% 
  group_by(cat, Ano) %>% 
  summarise(n = n_distinct(id)) %>% 
  group_by(Ano) %>% 
  mutate(p = n/sum(n))

journal_info_year %>% 
  ggplot(aes(x = Ano, y = p, fill = cat))+
  geom_col()

###find in OpenAlex####---------------------------

folder_path <- "data_update_2025/via bucket/openalex_matches_pt_update/"

file_list <- list.files(folder_path, pattern = "\\.parquet$", full.names = TRUE)

oa_lattes_match <- file_list %>%
  lapply(read_parquet) %>%
  bind_rows() %>% 
  unique()

set.seed(1234)

oa_lattes_match <- oa_lattes_match %>% 
  group_by(DOI) %>%
  slice_sample(n = 1) %>%
  ungroup()

n_distinct(oa_lattes_match$DOI)
# 1840324
table(oa_lattes_match$domain)

write_parquet(oa_lattes_match,"data_update_2025/oa_lattes_match.parquet")

oa_lattes_match <- read_parquet("data_update_2025/oa_lattes_match.parquet")

#check 
meta_lattes <- read_parquet("data_update_2025/update_meta_lattes.parquet") %>% 
  rename("id" = "Titulo") 

missing <- meta_lattes %>% 
  filter(!is.na(DOI)) %>% 
  filter(!DOI %in% oa_lattes_match$DOI)
#ok

### title match ##

folder_path <- "data_update_2025/via bucket/openalex_matches_title/"

file_list <- list.files(folder_path, pattern = "\\.parquet$", full.names = TRUE)

oa_lattes_match_title <- file_list %>%
  lapply(read_parquet) %>%
  bind_rows() %>% 
  unique()

set.seed(1234)

oa_lattes_match_title <- oa_lattes_match_title %>% 
  group_by(Titulo,Ano) %>%
  slice_sample(n = 1) %>%
  ungroup()

n_distinct(oa_lattes_match_title$Titulo)
# 1828219

write_parquet(oa_lattes_match_title,"data_update_2025/oa_lattes_title_match.parquet")

# fewer!

### in openalex labelled as Brazil

folder_path <- "data_update_2025/via bucket/oa_not_lattes/"

file_list <- list.files(folder_path, pattern = "\\.parquet$", full.names = TRUE)

oa_not_lattes_full <- file_list %>%
  lapply(read_parquet) %>%
  bind_rows() %>% 
  unique()

oa_not_lattes <- oa_not_lattes_full %>% 
  select(openalex_id,title,doi) %>%
  unique() 


#compare to the ones found by doi

oa_lattes_match <- read_parquet("data_update_2025/oa_lattes_match.parquet")

oa_not_lattes <- oa_not_lattes %>% 
  mutate(matched = ifelse(openalex_id %in% oa_lattes_match$openalex_id, TRUE, FALSE))

#remove cases that are obviously not articles (for example, title: "Reviewers")
oa_not_lattes <- oa_not_lattes %>%
  mutate(char = nchar(title)) %>%
  filter(char > 15) %>%
  select(-char,-title) %>% 
  unique()

oa_not_lattes %>% 
  group_by(matched) %>% 
  summarise(n = n_distinct(openalex_id))

# 659723 cases

oa_not_lattes %>% 
  filter(!matched) %>% 
  mutate(has_doi = ifelse((is.na(doi)|nchar(doi) < 5),FALSE,TRUE)) %>% 
  group_by(has_doi) %>% 
  summarise(n = n_distinct(openalex_id))

#they all have dois

explore <- oa_not_lattes %>% 
  filter(!matched) %>% 
  select(openalex_id) %>% 
  left_join(oa_not_lattes_full, by = "openalex_id")

write_parquet(explore,"data_update_2025/oa_not_lattes_explore.parquet")

rm(oa_lattes_match,oa_not_lattes,oa_not_lattes_full)

######all missing OpenAlex cases#####

explore <- read_parquet("data_update_2025/oa_not_lattes_explore.parquet")

inst_freq <- explore %>% 
  group_by(institution_type,institution_name) %>% 
  summarise(n = n_distinct(openalex_id))


check <- explore %>% 
  filter(institution_name == "Universidade de São Paulo")


explore %>% 
  group_by(publication_year) %>% 
  summarise(n = n_distinct(openalex_id))

explore %>% 
  group_by(domain) %>% 
  summarise(n = n_distinct(openalex_id))

#are authors there?



author_level <- explore %>%
  select(author_id, author_name) %>% 
  unique()


# write_parquet(find_authors,"data_update_2025/oa_not_lattes_find_authors.parquet")

#use alysson's match instead
author_match <- read_parquet("data_update_2025/openalex_lattes_authors.parquet")

author_level <- author_level %>% 
  mutate(in_lattes = ifelse(author_id %in% author_match$author_id, TRUE,FALSE))

table(author_level$in_lattes)
# FALSE   TRUE 
# 500627 254654 

#paper level
explore <- explore %>% 
  mutate(author_in_lattes = ifelse(author_id%in% author_match$author_id, TRUE,FALSE)) %>% 
  group_by(openalex_id) %>% 
  mutate(some_lattes_author = ifelse(any(author_in_lattes), TRUE,FALSE)) %>% 
  ungroup()

write_parquet(explore,"data_update_2025/oa_not_lattes_author_match.parquet")

explore %>% 
  group_by(some_lattes_author) %>% 
  summarise(n = n_distinct(openalex_id))

# lattes_author      n
# 1 FALSE         186083
# 2 TRUE          473640

#473640/659723 = 0.7179377

set.seed(1234)

check_sample <- explore %>% 
  filter(author_in_lattes) %>% 
  filter(!is.na(domain)) %>% 
  group_by(domain) %>% 
  slice_sample(n = 25)

check_sample %>% 
  writexl::write_xlsx("results_update_2025/sample_matched_authors_not_dois.xlsx")


set.seed(1234)

check_sample_b <- explore %>% 
  filter(!some_lattes_author) %>% 
  filter(!is.na(domain)) %>% 
  group_by(domain) %>% 
  slice_sample(n = 25)

check_sample_b %>% 
  writexl::write_xlsx("results_update_2025/sample_unmatched_authors_not_dois.xlsx")


###find in WoS####

final_doi_list <- meta_lattes %>%
  filter(!is.na(DOI)) %>% 
  select("doi" = "DOI") %>% 
  filter(!is.na(doi)) %>% 
  mutate(doi = str_squish(doi)) %>% 
  mutate(doi = sub("^doi:", "", doi)) %>% 
  mutate(doi = sub("^DOI:", "", doi)) %>% 
  mutate(doi = str_squish(doi)) %>% 
  unique()

write.csv(final_doi_list,"data_update_2025/doi_list_search_wos.csv",row.names = F)



wos_matches <- read_delim("data_update_2025/lattes_wos_matches.csv", 
           delim = ";", escape_double = FALSE, 
           col_names = c("DOI", "OST_BK","ESpecialite","EDiscipline"), 
           trim_ws = TRUE, na = c("NULL","")) %>% 
  mutate(DOI = gsub('"', "", DOI)) %>% 
  distinct(DOI, .keep_all = T)



###title match in wos ##

title_year_list <- meta_lattes %>% 
  mutate(Title = str_squish(id)) %>% 
  mutate(c = nchar(Title)) %>% 
  filter(c > 10) %>% 
  select(-c) %>% 
  select(Title,Ano) %>% 
  unique()

write.csv(title_year_list,"data_update_2025/title_list_search_wos.csv",row.names = F, fileEncoding = "utf-8")

wos_matches_title <- read_delim("data_update_2025/wos_title_matches.csv", 
                          delim = ";", escape_double = FALSE, 
                          trim_ws = TRUE, na = c("NULL","")) 
#fewer!

### in wos labelled as Brazil

wos_not_lattes_full <- read_delim("data_update_2025/wos_not_lattes.csv", 
                          delim = ";", escape_double = FALSE, 
                          col_names = c("OST_BK","Title","year","ESpecialite","EDiscipline","First_Name","Last_Name","Institution", "Identifier"), 
                          trim_ws = TRUE, na = c("NULL",""))

set.seed(1234)

wos_not_lattes <- wos_not_lattes_full %>% 
  group_by(OST_BK) %>%
  slice_sample(n = 1) %>%
  ungroup()

wos_not_lattes <- wos_not_lattes %>% 
  mutate(matched = ifelse(OST_BK %in% wos_matches$OST_BK, TRUE, FALSE))

table(wos_not_lattes$matched)
# 26,988 not matched

freqs <- wos_not_lattes %>% 
  filter(!matched) %>% 
  group_by(EDiscipline) %>% 
  summarise(n = n_distinct(OST_BK)) %>% 
  mutate(p = (n/sum(n))*100) %>% 
  select(-n) %>% 
  mutate(p = round(p,0))

sum(freqs$p)

freqs$p[freqs$EDiscipline == "Arts"] <- freqs$p[freqs$EDiscipline == "Arts"] +1

check_sample <- data.frame("OST_BK" = numeric(),"EDiscipline"= character(),matched = logical())

for (i in seq(length(unique(freqs$EDiscipline)))) {
  
  set.seed(1234)
  
  check_samplei <- wos_not_lattes %>% 
    filter(!matched) %>% 
    select(OST_BK,EDiscipline,matched) %>% 
    filter(EDiscipline ==freqs$EDiscipline[i]) %>% 
    slice_sample(n = freqs$p[i])
  
  check_sample <- bind_rows(check_sample,check_samplei)
  
}


set.seed(1234)

explore <- check_sample %>% 
  select(OST_BK) %>% 
  left_join(wos_not_lattes_full, by = "OST_BK") %>% 
  group_by(OST_BK) %>%
  slice_sample(n = 1) %>%
  ungroup()

explore %>% 
  writexl::write_xlsx("results_update_2025/sample_unmatched_wos.xlsx")


###compile manual validation results#####

stats_sample_matched_oa <- readxl::read_excel("results_update_2025/sample_matched_authors_not_dois_manual.xlsx")

matched_oa <- stats_sample_matched_oa %>% 
  group_by(in_lattes,`detail of retrieval`) %>% 
  summarise(n = n_distinct(openalex_id)) %>% 
  ungroup() %>% 
  mutate(p = n/sum(n))


stats_sample_unmatched_oa <- readxl::read_excel("results_update_2025/sample_unmatched_authors_not_dois_manual.xlsx")

unmatched_oa <- stats_sample_unmatched_oa %>% 
  group_by(`brazilian affiliation`,in_lattes,`detail of retrieval`) %>% 
  summarise(n = n_distinct(openalex_id)) %>% 
  ungroup() %>% 
  mutate(p = n/sum(n))


stats_sample_wos <- readxl::read_excel("results_update_2025/sample_unmatched_wos_manual.xlsx")

wos <- stats_sample_wos %>% 
  group_by(`brazilian affiliation`,in_lattes,`detail of retrieval`) %>% 
  summarise(n = n_distinct(OST_BK)) %>% 
  ungroup() %>% 
  mutate(p = n/sum(n))

writexl::write_xlsx(list("wos" = wos,
                         "matched_oa" = matched_oa,
                         "unmatched_oa" = unmatched_oa),
                    "results_update_2025/manual_validation_results.xlsx")
