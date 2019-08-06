library(tidyverse)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
library(rebus)
library(keras)
library(tensorflow)
library(tm)
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
#--------------
# remove digits
num_pattern <- one_or_more(DGT)
# remove stopwords
DATA %>%
  mutate(text = map_chr(text, str_remove_all, pattern = num_pattern ) ) %>%
  mutate(header = map(text, str_split, pattern = "\n\n") %>%
           map(~.x[[1]][[1]] )) %>%
  mutate(header = as.character(header) ) %>%
  mutate(main = map(text, str_split, pattern = "\n\n")) -> elo #%>%
           #map(~.x[[1]][[]] ) ) -> elo

elo$main[[1]] %>% unlist() -> one_article
one_article %>% length() -> length_paragraph




         main = map(text, str_split, pattern = "\n\n") %>%
           map(~.x[[1]])) -> elo
elo %>%
  mutate(header = as.character(header) )

  
  DATA %>%
    mutate(text = map_chr(text, str_remove_all, pattern = num_pattern ) ) %>%
    separate(text, "\n\n")



  mutate(header = toString(header))-> elo
#-----------
elo$header_help[[1]] %>% map(~.x[[1]])
#----------
  unnest_tokens(word, text) %>%
  anti_join(get_stopwords()) %>%
  nest(-name, -value, -category, -path, -in_train) %>%
  



#------------
maxlenth <- 100
maxwords <- 1e4

#-----
DATA$text[[1]] %>% str_replace_all(num_pattern, " ") %>%
  keras::text_tokenizer(num_words = 10) %>%
  keras::fit_text_tokenizer()

set.seed(1)
DATA %>%
  group_by(category) %>%
  sample_n(1) -> sample_data
sample_data %>%
  ungroup() %>%
  tidytext::unnest_tokens(output = word, text) %>% view()
#------------
sample_data$text[1] %>% keras::text_tokenizer(num_words = maxwords) -> elo

elo %>% text_to_word_sequence()

tf$Session() -> sesja
sesja$run()
#--------------
tokenize_text <- function(zbior, 
                          max_len = 100,
                          max_words = 10000){
  tokenizer <- text_tokenizer(num_words = max_words,
                              filters = "!\"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n") %>%
    fit_text_tokenizer(zbior$text)
  
  sekwencja <- texts_to_sequences(tokenizer, zbior$text )
  word_index <- tokenizer$word_index
  cat("Found", length(word_index), "unique tokens.\n")
  data <- pad_sequences(sekwencja, maxlen = max_len)
  return(data)
}

sample_data %>% tokenize_text()




