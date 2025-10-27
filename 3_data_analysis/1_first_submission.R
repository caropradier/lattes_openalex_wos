library(tidyverse)
library(arrow)
library(readr)
library(viridis)
library(wesanderson)
library(plotly)
library(htmlwidgets)
library(ggrepel)
library(patchwork)
options(scipen = 9999)


###load data#####

outlier_topics <- c(637,703,193,494)

document_topic <- read_parquet("job_outputs/merged_model_update/merged_document_topics.parquet") %>% 
  filter(!Topic %in%outlier_topics)

#improve automatic labels
topic_info <- read_csv("job_outputs/merged_model_update/labeled_topic_info.csv") %>% 
  mutate(Label = paste0(Topic, " - ", Label)) %>% 
  filter(!Topic %in%outlier_topics) %>% 
  mutate(Label = case_when(Topic == 149 ~ "149 - Gastric Procedures",
                           Topic == 638 ~ "638 - Bovine Mastitis Pathogens",
                           Topic == 562 ~ "562 - Catfish Taxonomy",
                           Topic == 230 ~"230 - Chagas Disease Treatment",
                           Topic == 234 ~"234 - Triatomine insects (Chagas)",
                           Topic == 552 ~"552 - Inclusive Education",
                           Topic == 679 ~"679 - Disability in Brazil",
                           Topic == 519 ~"519 - Environmental Education in Brazil",
                           Topic == 653 ~"653 - Municipal Family Health Policies",
                           Topic == 188 ~"188 - Gender Equality in Brazil",
                           Topic == 160 ~"160 - Population Genetics",
                           Topic == 46 ~"46 - Brazilian Higher Education",
                           Topic == 571 ~"571 - Student Mental Health",
                           Topic == 349 ~"349 - Corn Nitrogen Fertilization",
                           Topic == 621 ~"621 - Occupational Pesticide Exposure",
                           Topic == 394 ~"394 - Physical Exercise",
                           Topic == 672 ~"672 - Physical Education in Brazil",
                           Topic == 565 ~"565 - Rural Settlement Development",
                           Topic == 623 ~"623 - Regional Rural Development",
                           TRUE ~ Label)) 

coordinates <- read_csv("job_outputs/merged_model_update/coordinates_tsne.csv")
coordinates$topic <- seq.int(nrow(coordinates)) -1
coordinates <- coordinates %>% 
  filter(!topic %in%outlier_topics)

meta_lattes <- read_parquet("data_update_2025/update_meta_lattes.parquet") %>% 
  rename("id" = "Titulo") 

####main variables data#####

#Languages

language <- read_parquet("data_update_2025/update_language_inferences.parquet") %>% 
  select(-Confidence, -Titulo, -DOI)

language_topic <- document_topic %>% 
  left_join(language, by = "id") %>% 
  group_by(Topic) %>% 
  summarise(n_en = n_distinct(id[Language == "en"]),
            n = n_distinct(id)) %>% 
  mutate(p = n_en/n)

#OpenAlex coverage

oa_lattes_match <- read_parquet("data_update_2025/oa_lattes_match.parquet") 

oa_topic <- document_topic %>% 
  left_join(meta_lattes %>% select(id,DOI),by = "id") %>% 
  left_join(oa_lattes_match, by = "DOI") %>% 
  group_by(Topic) %>% 
  summarise(n_oa = n_distinct(id[!is.na(openalex_id)]),
            n = n_distinct(id)) %>% 
  mutate(p = n_oa/n)

# Discipline

dis_topic <- document_topic %>% 
  left_join(meta_lattes %>% select(id,DOI),by = "id") %>% 
  left_join(oa_lattes_match, by = "DOI") %>% 
  unique()

disciplines <- dis_topic %>% 
  filter(!is.na(domain)) %>% 
  group_by(Topic, domain) %>% 
  summarise(n = n_distinct(id)) %>% 
  group_by(Topic) %>% 
  mutate(p = n/sum(n)) %>% 
  slice_max(order_by = p, n =1) %>% 
  mutate(domain = ifelse(p < .55, "Multidisciplinary", domain)) %>% 
  select(-n,-p) %>% unique()

#WoS coverage

wos_matches <- read_delim("data_update_2025/lattes_wos_matches.csv", 
                          delim = ";", escape_double = FALSE, 
                          col_names = c("DOI", "OST_BK","ESpecialite","EDiscipline"), 
                          trim_ws = TRUE, na = c("NULL","")) %>% 
  mutate(DOI = gsub('"', "", DOI)) %>% 
  distinct(DOI, .keep_all = T)

wos_topic <- document_topic %>% 
  left_join(meta_lattes %>% select(id,DOI),by = "id") %>% 
  left_join(wos_matches, by = "DOI") %>% 
  group_by(Topic) %>% 
  summarise(n_wos = n_distinct(id[!is.na(OST_BK)]),
            n = n_distinct(id)) %>% 
  mutate(p = n_wos/n)

#Gender

gender_frac <- read_parquet("data_update_2025/update_gender_fractional.parquet") %>% 
  rename("id" = "Titulo")

topic_gender <- document_topic %>% 
  left_join(gender_frac, by = "id") %>% 
  group_by(Topic) %>% 
  summarise(Women = mean(Women,na.rm = T),
            n = n_distinct(id))


###indexation and language comparison####


lang_index <- coordinates %>% 
  left_join(disciplines, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic")) %>% 
  left_join(oa_topic %>% select(Topic, "p_oa" = "p"), by = c("topic" = "Topic")) %>% 
  left_join(wos_topic %>% select(Topic, "p_wos" = "p"), by = c("topic" = "Topic")) %>% 
  left_join(language_topic %>% select(Topic, "p_en" = "p"), by = c("topic" = "Topic")) %>% 
  mutate(p_en = ifelse(is.na(p_en),0,p_en)) %>% 
  mutate(p_oa = ifelse(is.na(p_oa),0,p_oa)) %>% 
  mutate(p_wos = ifelse(is.na(p_wos),0,p_wos)) 

labs <- lang_index %>% 
  rename("OpenAlex" = "p_oa") %>% 
  rename("Web of Science" = "p_wos") %>% 
  pivot_longer(c("OpenAlex","Web of Science"), names_to = "ind",values_to = "p") %>% 
  select(ind) %>% unique() %>% 
  mutate(cor_lab = case_when(ind == "OpenAlex" ~cor(lang_index$p_oa, lang_index$p_en, method = "spearman"),
                             ind == "Web of Science" ~cor(lang_index$p_wos, lang_index$p_en, method = "spearman"))) %>% 
  mutate(cor_lab = paste0("Spearman\ncorrelation:\n",round(cor_lab,4))) %>% 
  mutate(x = .15, y = .94)

lang_index %>% 
  rename("OpenAlex" = "p_oa") %>% 
  rename("Web of Science" = "p_wos") %>% 
  pivot_longer(c("OpenAlex","Web of Science"), names_to = "ind",values_to = "p") %>% 
  ggplot(aes(x = p_en, y = p, color = domain, size = Count))+
  geom_point(alpha = .7)+
  geom_smooth(color = "black", size = .5, fill="black")+
  geom_text(data = labs, aes(label = cor_lab, x = x, y = y), inherit.aes = FALSE,size =3)+
  geom_hline(yintercept = .25, alpha = .5)+
  geom_hline(yintercept = .5, alpha = .5)+
  geom_hline(yintercept = .75, alpha = .5)+
  geom_vline(xintercept = .5, alpha = .5)+
  theme_minimal()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = RColorBrewer::brewer.pal(name = "Paired", n =6)[c(1,2,5,4,6)], labels = function(x) str_wrap(x,20))+
  scale_size_continuous(range = c(1,8))+
  facet_wrap(~ind)+
  scale_y_continuous(labels = function(x) paste0(x*100,"%"),limits = c(0,1))+
  scale_x_continuous(labels = function(x) paste0(x*100,"%"),limits = c(0,1))+
  guides(size = "none",
         color =guide_legend(override.aes = list(size = 3))
         )+
  theme(legend.position = "top")+
  labs(y = "% indexed", x= "% written in English", color = "")

ggsave("results_update_2025/paper_draft/rel_english_indexation_facet.png", bg = "white", width = 8, height = 6) 



cor(lang_index$p_oa, lang_index$p_en, method = "spearman")
cor(lang_index$p_wos, lang_index$p_en, method = "spearman")

cor.test(lang_index$p_oa, lang_index$p_en, method = "spearman")
cor.test(lang_index$p_wos, lang_index$p_en, method = "spearman")


####gender areas#####


m_w <- weighted.mean(x = topic_gender$Women, w = topic_gender$n)
m_e <- weighted.mean(x = language_topic$p, w = language_topic$n)

clusters_alt <- coordinates %>% 
  left_join(disciplines, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic")) %>% 
  left_join(topic_gender, by = c("topic" = "Topic")) %>% 
  left_join(language_topic %>% select(Topic, "p_en" = "p"), by = c("topic" = "Topic")) %>% 
  mutate(p_en = ifelse(is.na(p_en),0,p_en)) %>%
  mutate(Women = ifelse(is.na(Women),0,Women)) %>%
  mutate(cluster = case_when(Women >= m_w & p_en >= m_e ~ "1. High women authorship - High % English",
                             Women >= m_w & p_en < m_e ~ "2. High women authorship - Low % English",
                             Women < m_w & p_en >= m_e ~ "3. Low women authorship - High % English",
                             Women < m_w & p_en < m_e ~ "4. Low women authorship - Low % English",
                             TRUE ~ "3. Intermediate")) 

table(clusters_alt$cluster)


cluster_palette <- c("#946DC0" ,"gold","#66A61E", "#FC8D62")

interesting <- c(212,393,#multidis cluster in physical
                 115,282,69,5,700,515,175,201,147,481,111,366,#feminized soc sci
                 240,272,64,210,711,622,595,405,736,#masculinized soc sci
                 229,554,152,206,332,16,#multidis cluster in life
                 37,96,83,197,521,678,79,161,443,134,209,66,95,4,694,467,48,381,448,13,#eng-women-health
                 471,411,110,359,317,123,496,624,153,186,#eng-men-health
                 92,370,113,34,565,263,#low english life
                 165,297,363,546,2,245,486,35,491,374,314, #english life
                 98,232,614,215,367,290,662,20,276,461,357,372, #english physical
                 51,462,86,476,500 #women english physical
                 )



clusters_alt%>% 
  ggplot(aes(x=x,y=y, color = domain,size = Count
  ))+
  geom_point(alpha = .7)+
  stat_density_2d(data =clusters_alt , 
                  aes(fill = cluster,color = cluster), 
                  geom = "polygon", alpha = .1, 
                  #bins = 5
                  bins = 8
  )+
  geom_text_repel(data = subset(clusters_alt,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                      ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
                  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = c(cluster_palette,
                                "#A6CEE3", "#1F78B4", "#FB9A99", "#33A02C", "#E31A1C"), 
                     labels = function(x) str_wrap(x,20))+
  scale_fill_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3))
  )+
  theme(legend.box.margin = margin(0,0,3,0))

ggsave("results_update_2025/paper_draft/clusters_gender_labels.png", bg = "white", width = 8, height = 8) 


clusters_alt%>% 
  ggplot(aes(x=x,y=y, color = cluster,size = Count))+
  geom_point(alpha = .7)+
  geom_text_repel(data = subset(clusters_alt,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                  ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3),
                             nrow = 2))

ggsave("results_update_2025/paper_draft/annex_clusters_gender_labels.png", bg = "white", width = 8, height = 8) 


####oa gender areas -alt#####


m_w <- weighted.mean(x = topic_gender$Women, w = topic_gender$n)
m_oa <- weighted.mean(x = oa_topic$p, w = oa_topic$n)

clusters_alt_oa <- coordinates %>% 
  left_join(disciplines, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic")) %>% 
  left_join(topic_gender, by = c("topic" = "Topic")) %>% 
  left_join(oa_topic %>% select(Topic, p), by = c("topic" = "Topic")) %>% 
  mutate(p = ifelse(is.na(p),0,p)) %>%
  mutate(Women = ifelse(is.na(Women),0,Women)) %>%
  mutate(cluster = case_when(Women >= m_w & p >= m_oa ~ "1. High women authorship - High % OA indexing",
                             Women >= m_w & p < m_oa ~ "2. High women authorship - Low % OA indexing",
                             Women < m_w & p >= m_oa ~ "3. Low women authorship - High % OA indexing",
                             Women < m_w & p < m_oa ~ "4. Low women authorship - Low % OA indexing",
                             TRUE ~ "3. Intermediate")) 

table(clusters_alt_oa$cluster)


clusters_alt_oa%>% 
  ggplot(aes(x=x,y=y, color = domain,size = Count
  ))+
  geom_point(alpha = .7)+
  stat_density_2d(data =clusters_alt_oa , 
                  aes(fill = cluster,color = cluster), 
                  geom = "polygon", alpha = .1, 
                  bins = 8
  )+
  geom_text_repel(data = subset(clusters_alt_oa,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                  ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = c(cluster_palette,
                                "#A6CEE3", "#1F78B4", "#FB9A99", "#33A02C", "#E31A1C"), 
                     labels = function(x) str_wrap(x,20))+
  scale_fill_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3))
  )+
  theme(legend.box.margin = margin(0,0,3,0))

ggsave("results_update_2025/paper_draft/oa_clusters_gender_labels.png", bg = "white", width = 8, height = 8) 


clusters_alt_oa%>% 
  ggplot(aes(x=x,y=y, color = cluster,size = Count))+
  geom_point(alpha = .7)+
  geom_text_repel(data = subset(clusters_alt_oa,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                  ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3),
                             nrow = 2))

ggsave("results_update_2025/paper_draft/annex_oa_clusters_gender_labels.png", bg = "white", width = 8, height = 8) 


####wos gender areas -alt#####


m_w <- weighted.mean(x = topic_gender$Women, w = topic_gender$n)
m_wos <- weighted.mean(x = wos_topic$p, w = wos_topic$n)

clusters_alt_wos <- coordinates %>% 
  left_join(disciplines, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic")) %>% 
  left_join(topic_gender, by = c("topic" = "Topic")) %>% 
  left_join(wos_topic %>% select(Topic, p), by = c("topic" = "Topic")) %>% 
  mutate(p = ifelse(is.na(p),0,p)) %>%
  mutate(Women = ifelse(is.na(Women),0,Women)) %>%
  mutate(cluster = case_when(Women >= m_w & p >= m_wos ~ "1. High women authorship - High % WoS indexing",
                             Women >= m_w & p < m_wos ~ "2. High women authorship - Low % WoS indexing",
                             Women < m_w & p >= m_wos ~ "3. Low women authorship - High % WoS indexing",
                             Women < m_w & p < m_wos ~ "4. Low women authorship - Low % WoS indexing",
                             TRUE ~ "3. Intermediate")) 

table(clusters_alt_wos$cluster)





clusters_alt_wos%>% 
  ggplot(aes(x=x,y=y, color = domain,size = Count
  ))+
  geom_point(alpha = .7)+
  stat_density_2d(data =clusters_alt_wos , 
                  aes(fill = cluster,color = cluster), 
                  geom = "polygon", alpha = .1, 
                  bins = 8
  )+
  geom_text_repel(data = subset(clusters_alt_wos,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                  ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = c(cluster_palette,
                                "#A6CEE3", "#1F78B4", "#FB9A99", "#33A02C", "#E31A1C"), 
                     labels = function(x) str_wrap(x,20))+
  scale_fill_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3))
  )+
  theme(legend.box.margin = margin(0,0,3,0))

ggsave("results_update_2025/paper_draft/wos_clusters_gender_labels.png", bg = "white", width = 8, height = 8) 

clusters_alt_wos%>% 
  ggplot(aes(x=x,y=y, color = cluster,size = Count))+
  geom_point(alpha = .7)+
  geom_text_repel(data = subset(clusters_alt_wos,topic %in%interesting),
                  aes(x = x, y = y,label = str_wrap(sub(".* - ", "", Label),10)
                  ),
                  size = 3, fontface = 'bold', color = "black",alpha = 1,
                  segment.alpha = 1,
                  force = 15,
                  max.overlaps = Inf,
                  min.segment.length = 0.01,
                  segment.size = 0.5,
                  nudge_y = .3,
                  seed = 555
  )+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_manual(values = cluster_palette)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "",shape = "")+
  guides(size = "none",
         fill = "none", shape = "none",
         color =guide_legend(override.aes = list(linewidth = 3,size = 3),
                             nrow = 2))

ggsave("results_update_2025/paper_draft/annex_wos_clusters_gender_labels.png", bg = "white", width = 8, height = 8) 



###descriptive####

order_disciplines <- c("Physical Sciences","Life Sciences","Health Sciences" ,"Multidisciplinary" ,"Social Sciences"  )

full_full <- document_topic %>% 
  select(-Titulo) %>% 
  left_join(meta_lattes %>% select(id,DOI),by = "id") %>% 
  left_join(gender_frac, by = "id") %>% 
  filter(!is.na(Women)) %>% 
  left_join(disciplines, by= c("Topic")) %>% 
  left_join(language, by = "id") %>% 
  mutate(Language = ifelse(Language == "en", "English", "Other")) %>% 
  mutate(domain = factor(domain,
                             levels = order_disciplines)) %>% 
  mutate(in_openalex = ifelse(DOI %in% oa_lattes_match$DOI, "Indexed","Not indexed"),
         in_wos = ifelse(DOI %in% wos_matches$DOI,  "Indexed","Not indexed"))


####introduction#####

n_pubs <- n_distinct(full_full$id)


a <- bind_rows(
full_full %>% 
  summarise(OpenAlex = sum(Women[in_openalex == "Indexed"])/sum(Women),
            WoS = sum(Women[in_wos == "Indexed"])/sum(Women),
            English = sum(Women[Language == "English"])/sum(Women)
  ) %>% 
  pivot_longer(everything(),names_to = "ind", values_to = "value")%>% 
  mutate(gender = "Women"),
full_full %>% 
  summarise(OpenAlex = sum(Men[in_openalex == "Indexed"])/sum(Men),
            WoS = sum(Men[in_wos == "Indexed"])/sum(Men),
            English = sum(Men[Language == "English"])/sum(Men)
  ) %>% 
  pivot_longer(everything(),names_to = "ind", values_to = "value")%>% 
  mutate(gender = "Men")
) %>% 
  mutate(ind = case_when(ind == "OpenAlex" ~ "% indexed in OpenAlex",
                         ind == "WoS" ~ "% indexed in WoS",
                         ind == "English" ~ "% written in English")) %>% 
  mutate(ind = factor(ind, levels = rev(c("% indexed in WoS",
                                          "% indexed in OpenAlex",
                                          "% written in English")
  ))) %>% 
  ggplot(aes(x = gender, y =value,fill = ind))+
  geom_col(position = "dodge")+
  geom_text(aes(label = paste0(round(value*100,1), "%"), group = ind),position = position_dodge(width = .9), hjust = -0.1,size =3)+
  theme_minimal()+
  theme(legend.position = c(.75,.6))+
  scale_y_continuous(labels = function(x) paste0(x*100,"%"), limits = c(0,1))+
  scale_x_discrete( labels = function(x) str_wrap(x,10))+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name="Paired", 12)[c(7,6,4)],labels = function(x) str_wrap(x,15))+
  #scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(1,3)])+
  labs(fill = "", y = "% Publications", x ="", title = "C")+
  coord_flip()+
  theme(panel.grid.minor = element_blank())+
  theme(panel.grid.major = element_blank())+
  guides(fill = guide_legend(reverse = TRUE, nrow = 1))

a

b <-  full_full %>% 
  mutate( indexation = case_when(
    in_openalex == "Indexed"& in_wos != "Indexed" ~"Only OpenAlex",
    in_wos == "Indexed"& in_openalex != "Indexed"~"Only WoS",
    in_wos == "Indexed" & in_openalex == "Indexed" ~"WoS & OpenAlex",
    in_openalex != "Indexed" & in_wos != "Indexed" ~"Non-indexed by WoS or OA"
  )) %>% 
  group_by(indexation) %>% 
  summarise(n = n_distinct(id)) %>% 
  mutate(p = n/sum(n)) %>% 
  mutate(c = "") %>% 
  mutate(indexation = factor(indexation, 
                             levels = rev(c( "Only WoS","WoS & OpenAlex","Only OpenAlex","Non-indexed by WoS or OA"))
  )) %>% 
  ggplot(aes(x = c, y = p, fill=indexation))+
  geom_col(position = "stack", width = .5)+
  geom_label(aes(label = paste0(round(p*100,2), "%"), 
                group = indexation
                ,fill = indexation
                 ,y = ifelse((p > .26|p<.2), p+.02, p-.07)
                ), 
            #position = position_stack(vjust = 0.5),
            nudge_x = .47,
            label.size = NA,
            label.r = unit(0, "lines"),
            show.legend = FALSE,
            #nudge_y = .01,
            #,fontface = "bold"
            alpha = .9,
            size =3)+
  theme_minimal()+
  theme(legend.position = "top")+
  scale_y_continuous(labels = function(x) paste0(x*100,"%"))+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name="Paired", 12)[c(9,6,3,4)])+
  scale_color_manual(values = RColorBrewer::brewer.pal(name="Paired", 12)[c(9,6,3,4)])+
  labs(fill = "Publication indexation", y = "% Publications", x ="", title = "A")+
  coord_flip()+
  theme(panel.grid.minor = element_blank())+
  theme(panel.grid.major = element_blank())+
  guides(fill = guide_legend(reverse=T), color = "none")

b

##add estimation of upper bound to the text

corpus_a_size <- 473640
a_matched <- .41
a_not_covered <- .01 
max_additional_a <- corpus_a_size*(a_matched+a_not_covered)

corpus_b_size <- 186083
b_matched <- .26
b_not_covered <- .10 
max_additional_b <- corpus_b_size*(b_matched+b_not_covered)

corpus_c_size <- 26988
c_matched <- .58
c_not_covered <- .26 
max_additional_c <- corpus_c_size*(c_matched+c_not_covered)


original_table <- full_full %>% 
  mutate( indexation = case_when(
    in_openalex == "Indexed"& in_wos != "Indexed" ~"Only OpenAlex",
    in_wos == "Indexed"& in_openalex != "Indexed"~"Only WoS",
    in_wos == "Indexed" & in_openalex == "Indexed" ~"WoS & OpenAlex",
    in_openalex != "Indexed" & in_wos != "Indexed" ~"Non-indexed by WoS or OA"
  )) %>% 
  group_by(indexation) %>% 
  summarise(n = n_distinct(id)) %>% 
  mutate(p = n/sum(n)) 


#max proportion considering:
#1.Matched to Lattes using title, authors, and publication year 
#2.Eligible journal articles not covered by the Lattes database

#For OpenAlex
(sum(original_table$n[original_table$indexation %in% 
                       c("Only OpenAlex","WoS & OpenAlex")]) + max_additional_a + max_additional_b)/n_pubs

#For WoS
(sum(original_table$n[original_table$indexation %in% 
                       c("Only WoS","WoS & OpenAlex")]) + 
  max_additional_c)/n_pubs

####


c <- full_full %>% 
  summarise(OpenAlex = n_distinct(id[in_openalex == "Indexed" &Language == "English"])/n_distinct(id[in_openalex == "Indexed"]),
            WoS = n_distinct(id[in_wos == "Indexed"&Language == "English"])/n_distinct(id[in_wos == "Indexed"]),
            #"WoS & OpenAlex" = n_distinct(id[in_wos == "Indexed" & in_openalex == "Indexed"&Language == "English"])/n_distinct(id[in_wos == "Indexed" & in_openalex == "Indexed"]),
            "Non-indexed by WoS or OA" = n_distinct(id[in_wos != "Indexed" & in_openalex != "Indexed"&Language == "English"])/n_distinct(id[in_wos != "Indexed" & in_openalex != "Indexed"])
           
  ) %>% 
  pivot_longer(everything(),names_to = "indexation", values_to = "p")  %>% 
  ggplot(aes(x = reorder(indexation,-p), y = p, fill=indexation))+
  geom_col()+
  geom_text(aes(label = paste0(round(p*100,1), "%"), group = indexation), hjust = -0.1,size =3)+
  theme_minimal()+
  theme(legend.position = "none")+
  scale_x_discrete(labels = function(x) str_wrap(x,10))+
  scale_y_continuous(labels = function(x) paste0(x*100,"%"),limits = c(0,1))+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name="Paired", 12)[c(9,6,4,3)])+
  theme(panel.grid.minor = element_blank())+
  theme(panel.grid.major = element_blank())+
  labs(#x = "Publication indexation", 
        x = "",
       y = "% written in English", title = "B")+
  coord_flip()

c

b/c/a+ plot_layout(heights = c(.2,.4,.4))
  
ggsave("results_update_2025/paper_draft/comp_eng_indexation.png", bg = "white", width = 8, height = 6)


####gender context - over/underrep#####

order_disciplines <- c("Physical Sciences","Life Sciences","Health Sciences" ,"Multidisciplinary" ,"Social Sciences"  )


ind_over_under <- full_full %>% 
  group_by(in_openalex,domain) %>% 
  summarise(Men = sum(Men),
            Women = sum(Women)) %>% 
  pivot_longer(c("Men", "Women"), names_to = "gender", values_to = "n") %>% 
  group_by(domain,gender) %>% 
  mutate(p =n/sum(n))  %>% 
  select(-n) %>% 
  left_join(
    full_full %>% 
      group_by(in_openalex,domain) %>% 
      summarise(n = n_distinct(id)) %>% 
      group_by(domain) %>% 
      mutate(p_norm =n/sum(n)) %>% 
      select(-n)
    
    ,by = c('in_openalex' ,'domain'))%>% 
  mutate(x = p/p_norm-1) %>% 
  filter(in_openalex == "Indexed") %>% 
  ggplot(aes(x = x, y = domain, fill = gender)) + 
  geom_col(position = "dodge")+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(1,3)])+
  theme_minimal()+
  theme(legend.position = "top")+
  scale_y_discrete( labels = function(x) str_wrap(x,10))+
  scale_x_continuous(limits = c(-.3,.3))+
  labs(x = "Ratio between observed and expected \n% of publications indexed in OpenAlex", fill = ""
       #, y = "Discipline"
       , y = ""
       ,title = "E"
  )+
  theme(panel.grid.minor = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))

ind_over_under

ind_over_under_wos <- full_full %>% 
  group_by(in_wos,domain) %>% 
  summarise(Men = sum(Men),
            Women = sum(Women)) %>% 
  pivot_longer(c("Men", "Women"), names_to = "gender", values_to = "n") %>% 
  group_by(domain,gender) %>% 
  mutate(p =n/sum(n))  %>% 
  select(-n) %>% 
  left_join(
    full_full %>% 
      group_by(in_wos,domain) %>% 
      summarise(n = n_distinct(id)) %>% 
      group_by(domain) %>% 
      mutate(p_norm =n/sum(n)) %>% 
      select(-n)
    
    ,by = c('in_wos' ,'domain'))%>% 
  mutate(x = p/p_norm-1) %>% 
  filter(in_wos == "Indexed") %>% 
  ggplot(aes(x = x, y = domain, fill = gender)) + 
  geom_col(position = "dodge")+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(1,3)])+
  theme_minimal()+
  theme(legend.position = "top")+
  scale_y_discrete( labels = function(x) str_wrap(x,10))+
  scale_x_continuous(limits = c(-.3,.3))+
  labs(x = "Ratio between observed and expected % \nof publications indexed in the WoS", fill = "", y = ""
       #y = "Discipline"
       ,title = "D"
  )+
  theme(panel.grid.minor = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))

ind_over_under_wos

eng_over_under <- full_full %>% 
  group_by(Language,domain) %>% 
  summarise(Men = sum(Men),
            Women = sum(Women)) %>% 
  pivot_longer(c("Men", "Women"), names_to = "gender", values_to = "n") %>% 
  group_by(domain,gender) %>% 
  mutate(p =n/sum(n))  %>% 
  select(-n) %>% 
  left_join(
    
    full_full %>% 
      group_by(Language,domain) %>% 
      summarise(n = n_distinct(id)) %>% 
      group_by(domain) %>% 
      mutate(p_norm =n/sum(n)) %>% 
      select(-n)
    
    ,by = c('Language' ,'domain'))%>% 
  mutate(x = p/p_norm-1) %>% 
  filter(Language == "English") %>% 
  ggplot(aes(x = x, y = domain, fill = gender)) + 
  geom_col(position = "dodge")+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(1,3)])+
  scale_y_discrete( labels = function(x) str_wrap(x,10))+
  scale_x_continuous(limits = c(-.3,.3))+
  theme_minimal()+
  theme(legend.position = "top")+
  labs(x = "Ratio between observed and expected % \nof publications written in English", fill = "", 
       y = "Discipline"
       ,title = "C"
  )+
  theme(panel.grid.minor = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))

eng_over_under

authorship <- full_full %>% 
  group_by(domain) %>% 
  summarise(Men = sum(Men),
            Women = sum(Women)) %>% 
  pivot_longer(c("Men", "Women"), names_to = "gender", values_to = "n") %>% 
  group_by(domain) %>% 
  mutate(p =n/sum(n)) %>% 
  ggplot(aes(x = p, y = domain, fill = gender))+
  geom_col()+
  geom_vline(aes(xintercept = .5),size = .2)+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(1,3)])+
  theme_minimal()+
  theme(legend.position = "top")+
  scale_x_continuous(labels = function(x) paste0(x*100,"%"))+
  scale_y_discrete( labels = function(x) str_wrap(x,10))+
  labs(fill = "", y = "Discipline", x = "% Authorship"
       ,title = "A"
  )+
  theme(panel.grid.minor = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))

authorship

ind_eng <- left_join(
  full_full %>% 
    group_by(Language,domain) %>% 
    summarise(n = n_distinct(id)) %>% 
    group_by(domain) %>% 
    mutate(p_norm =n/sum(n)) %>% 
    select(-n) %>% 
    filter(Language == "English") %>% 
    rename("% written in English"="p_norm") %>% 
    select(-Language)
  ,
  full_full %>% 
    group_by(in_openalex,domain) %>% 
    summarise(n = n_distinct(id)) %>% 
    group_by(domain) %>% 
    mutate(p_norm =n/sum(n)) %>% 
    select(-n) %>% 
    filter(in_openalex == "Indexed") %>% 
    rename("% indexed in OpenAlex"="p_norm") %>% 
    select(-in_openalex)
  ,by = "domain"
) %>% 
  left_join(
    full_full %>% 
      group_by(in_wos,domain) %>% 
      summarise(n = n_distinct(id)) %>% 
      group_by(domain) %>% 
      mutate(p_norm =n/sum(n)) %>% 
      select(-n) %>% 
      filter(in_wos == "Indexed") %>% 
      rename("% indexed in the WoS"="p_norm") %>% 
      select(-in_wos)
    ,by = "domain"
  ) %>% 
  pivot_longer(cols = -c("domain"), names_to = "ind", values_to = "value") %>% 
  mutate(ind = factor(ind, levels = rev(c("% indexed in the WoS",
                                      "% indexed in OpenAlex",
                                      "% written in English")
                                      ))) %>% 
  ggplot(aes(x = value, y = domain, fill = ind))+
  geom_col(position = "dodge")+
  #scale_fill_manual(values = RColorBrewer::brewer.pal(name = "Dark2",n=12)[c(5,2,6)],labels = function(x) str_wrap(x,15))+
  scale_fill_manual(values = RColorBrewer::brewer.pal(name="Paired", 12)[c(7,6,4)],labels = function(x) str_wrap(x,15))+
  theme_minimal()+
  theme(legend.position = "top")+
  scale_x_continuous(labels = function(x) paste0(x*100,"%"))+
  scale_y_discrete( labels = function(x) str_wrap(x,10))+
  labs(fill = "", y = "",
       #y = "Discipline", 
       x = "% Publications"
       ,title = "B"
  )+
  theme(panel.grid.minor = element_blank())+
  guides(fill = guide_legend(reverse = TRUE))

ind_eng

context <- (authorship + ind_eng)
over_under <- (eng_over_under + ind_over_under_wos + ind_over_under )

(context / over_under) 

ggsave("results_update_2025/paper_draft/over_underrep.png", bg = "white", width = 11, height = 8) 


###annex labeled space#####

n_show <- 25

tb_oa <- bind_rows(
  coordinates %>% 
    left_join(oa_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_max(order_by = p, n = n_show) 
  ,
  coordinates %>% 
    left_join(oa_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_min(order_by = p, n = n_show) 
) %>% 
  mutate(Label = sub(".* - ", "", Label))

coordinates %>% 
  left_join(oa_topic, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
  ggplot(aes(x=x,y=y, color = p,size = Count))+
  geom_point(alpha = .7)+
  ggrepel::geom_text_repel(data = tb_oa, 
                           aes(label = str_wrap(paste0(Label," (",round(p*100,1),"%)"),15) ,x=x,y=y), 
                           size = 3, fontface = 'bold',
                           alpha = 1, 
                           color = "black",
                           max.overlaps = Inf,
                           seed=555)+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_viridis(option ="F", begin =.2,  labels = function(x) paste0(x*100,"%"),limits = c(0,1))+
  scale_size_continuous(range = c(1,8))+
  labs(color = "% Publications in OpenAlex")+
  guides(size = "none")

ggsave("results_update_2025/paper_draft/annex_oa.png", bg = "white", width = 10, height = 8) 



tb_wos <- bind_rows(
  coordinates %>% 
    left_join(wos_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_max(order_by = p, n = n_show) 
  ,
  coordinates %>% 
    left_join(wos_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_min(order_by = p, n = n_show) 
) %>% 
  mutate(Label = sub(".* - ", "", Label))


coordinates %>% 
  left_join(wos_topic, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
  ggplot(aes(x=x,y=y, color = p,size = Count))+
  geom_point(alpha = .7)+
  theme_void()+
  theme(legend.position = "bottom")+
  ggrepel::geom_text_repel(data = tb_wos, aes(label = str_wrap(paste0(Label," (",round(p*100,1),"%)"),15) ,x=x,y=y), 
                           size = 3, fontface = 'bold',
                           alpha = 1, 
                           color = "black",
                           max.overlaps = Inf,
                           seed=555)+
  scale_color_viridis(option ="F", begin =.2,  labels = function(x) paste0(x*100,"%"),limits = c(0,1))+
  scale_size_continuous(range = c(1,8))+
  labs(color = "% Publications in the Web of Science")+
  guides(size = "none")

ggsave("results_update_2025/paper_draft/annex_wos.png", bg = "white", width = 10, height = 8) 


tb_en <- bind_rows(
  coordinates %>% 
    left_join(language_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_max(order_by = p, n = n_show) 
  ,
  coordinates %>% 
    left_join(language_topic, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_min(order_by = p, n = n_show) 
) %>% 
  mutate(Label = sub(".* - ", "", Label))

coordinates %>% 
  left_join(language_topic, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
  ggplot(aes(x=x,y=y, color = p,size = Count))+
  geom_point(alpha = .7)+
  theme_void()+
  theme(legend.position = "bottom")+
  ggrepel::geom_text_repel(data = tb_en, aes(label = str_wrap(paste0(Label," (",round(p*100,1),"%)"),15) ,x=x,y=y), 
                           size = 3, fontface = 'bold',
                           alpha = 1, 
                           color = "black",
                           max.overlaps = Inf,
                           seed=555)+
  scale_color_viridis(option = "G", labels = function(x) paste0(x*100,"%"))+
  scale_size_continuous(range = c(1,8))+
  labs(color = "% Publications in English")+
  guides(size = "none")

ggsave("results_update_2025/paper_draft/annex_lang.png", bg = "white", width = 10, height = 8) 


tb_w <- bind_rows(
  coordinates %>% 
    left_join(topic_gender, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic")) %>% 
    slice_max(order_by = Women, n = n_show) 
  ,
  coordinates %>% 
    left_join(topic_gender, by = c("topic" = "Topic")) %>% 
    left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
    slice_min(order_by = Women, n = n_show) 
) %>% 
  mutate(Label = sub(".* - ", "", Label))


coordinates %>% 
  left_join(topic_gender, by = c("topic" = "Topic")) %>% 
  left_join(topic_info %>% select(Topic,Label,Count), by = c("topic" = "Topic"))  %>% 
  ggplot(aes(x=x,y=y, color = Women,size = Count))+
  geom_point(alpha = .9)+
  theme_void()+
  theme(legend.position = "bottom")+
  scale_color_gradientn(colours = c("#E0ECF4","#BFD3E6", "#8C96C6", "#8C6BB1","#810F7C"  ,"#4D004B"), 
                        labels = function(x) paste0(x*100,"%"),
                        limits = c(0.1,1))+
  ggrepel::geom_text_repel(data = tb_w, aes(label = str_wrap(paste0(Label," (",round(Women*100,1),"%)"),15) ,x=x,y=y), size = 3, fontface = 'bold',
                           alpha = 1, 
                           color = "black",
                           max.overlaps = Inf,
                           seed=555)+
  scale_size_continuous(range = c(1,8))+
  labs(color = "% Women authorship")+
  guides(size = "none")

ggsave("results_update_2025/paper_draft/annex_women.png", bg = "white", width = 10, height = 8) 



###n checks methods######
nrow(document_topic)

folder_path <- "data_update_2025/via bucket/articles/"

file_list <- list.files(folder_path, pattern = "\\.parquet$", full.names = TRUE)

updated_lattes <- file_list %>%
  lapply(read_parquet) %>%
  bind_rows()

keep <- updated_lattes %>% 
  filter(Titulo %in% document_topic$id)

n_distinct(keep$IDLattes)

author_gender <- read_parquet("data_update_2025/update_lattes_inference.parquet") %>% 
  mutate(gender = case_when(gender == "M" ~ "Men",
                            gender == "F" ~ "Women",
                            TRUE ~ gender))


n_distinct(author_gender$author_id[author_gender$gender%in%c("Men","Women")])/n_distinct(keep$IDLattes)

rm(updated_lattes,keep,author_gender)

##previous corpus
prev_corpus <- read_parquet("job_outputs/merged_model_c/merged_document_topics.parquet")

check_overlap <- document_topic %>% 
  mutate(in_prev = ifelse(id %in% prev_corpus$Titulo, TRUE,FALSE))

check_overlap %>% 
  group_by(in_prev) %>% 
  summarise(n = n_distinct(id)) %>% 
  mutate(p = n/sum(n))

