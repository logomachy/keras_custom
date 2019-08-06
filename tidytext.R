library(tidyverse)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
setwd("bbc/")
#-------------
list.dirs() %>% stringr::str_remove_all("\\.") %>% stringr::str_remove_all("/") %>%
  .[ str_length(.) > 0 ] -> categories

data_load <- function(i, ...) {
  list.files(paste0(getwd(),"/", categories[i])) %>%
    enframe() %>%
    mutate(category = categories[i] ) %>%
    mutate(path = paste0(getwd(),"/", categories[i], "/", .$value) ) %>% 
    mutate(text = unlist(pmap(. , ~ with(list(...), read_file(path) )  )) ) %>%
    return(.)
}
lapply(1:length(categories), function(i) data_load(i) )  %>% bind_rows() -> DATA
caret::createDataPartition(DATA$category, p = 0.8, list = F) -> in_train 
DATA %<>% mutate(in_train = ifelse(name %in% in_train, T, F))
DATA %<>% mutate(text = map_chr(text, unlist )) 

set.seed(1)
DATA %>%
  group_by(category) %>%
  sample_n(1) -> sample_data
sample_data %>%
  ungroup() %>%
  tidytext::unnest_tokens(output = word, text) 
#------------