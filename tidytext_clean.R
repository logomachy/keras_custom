library(tidyverse)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
library(rebus)
library(keras)
library(tensorflow)
#library(tm)
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
###############
DATA %>%
  group_by(category) %>%
  sample_n(1) -> sample_data
##############
preprocess <- function(DATA, variable = "main") {
################
# HELPER FUNCTIONS
  remove_and_split <- function(DATA, ...) {
    pull_main_corpus  <- function(main) {
      main %>% 
        unlist() %>%
        .[-1] %>%
        toString() %>%
        return( . ) 
    }
    #---------
    num_pattern <- one_or_more(DGT) # regex for numbers
    
    DATA %>%
      mutate(text = map_chr(text, str_remove_all, pattern = num_pattern ) ) %>%
      mutate(header = map(text, str_split, pattern = "\n\n") %>%
               map(~.x[[1]][[1]] )) %>%
      mutate(header = as.character(header) ) %>%
      mutate(main = map(text, str_split, pattern = "\n\n")) %>%
      mutate(main = map_chr(main, pull_main_corpus)) %>%
      return(.)
  }
  collapse_word <- function(data) {
    data %>%
      pull() %>% paste0(collapse = " ") %>%
      return(.)
  }
  unnest_variable <- function(df, variable) {
    df %>%
      #rownames_to_column("id") 
      select(-text ,
             -!!sym(ifelse(
               variable == "header",
               "main",
               ifelse(variable == "main", "header", stop("BAD Colname")) )) ) %>%
      unnest_tokens(word,  !!sym(variable), to_lower = T) %>%
      anti_join(get_stopwords(language = "en"), by = "word") %>%
      #nest(-name, -value, -category, -path, -in_train) %>%
      return(.)
  }
################
  DATA %>% 
    remove_and_split() %>%
    rownames_to_column("id") %>% 
    select(-name, -value, -path) %>% 
    unnest_variable(variable) %>% 
    nest(-id, -category, -in_train) %>% 
    mutate(data = map_chr( data, collapse_word) ) %>%
    return(.)
}
tokenize_text <- function(zbior, 
                          max_len = 100,
                          max_words = 1e4){
  tokenizer <- text_tokenizer(num_words = max_words,
                              filters = "!\"#%&()$*+,-./:;<=>?@[\\]^_`{|}~\t\n") %>%
    fit_text_tokenizer(zbior$data)
  
  sekwencja <- texts_to_sequences(tokenizer, zbior$data )
  word_index <- tokenizer$word_index
  cat("##################\n")
  cat("Found", length(word_index), "unique tokens.\n")
  cat("##################\n")
  data <- pad_sequences(sekwencja, maxlen = max_len)
  return(list(data, word_index))
}
#############
DATA %>% 
  preprocess() %>%
  tokenize_text() -> elo
####################
# LOADING EMB
###################




