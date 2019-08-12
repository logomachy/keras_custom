library(tidyverse)
library(data.table)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
library(rebus)
library(keras)
#library(tensorflow) 
#-----------
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
DATA %<>% mutate(text = map_chr(text, unlist ))
set.seed(1)
sample(1:nrow(DATA), nrow(DATA)*0.2) -> test_index
DATA[-test_index, ] -> train_set
DATA[test_index, ] -> test_set
#--------------------------------------------
preprocess <- function(DATA,
                       variable = "main", no_folds = 10) {
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
  folds <- caret::createFolds(y = DATA$category , k = no_folds, list = F )
  
  DATA %>%
    remove_and_split() %>%
    rownames_to_column("id") %>%
    select(-name, -value, -path) %>%
    unnest_variable(variable) %>%
    #filter(stringr::str_length( word) >= 2)  %>%
    nest(-id, -category) %>%
    mutate(data = map_chr( data, collapse_word) ) %>%
    mutate(fold = folds) %>%
    return(.)
}
onehot_represenation <- function(jolo) {
  jolo %>% as_tibble() %>% select_if(is.factor) -> factor_var
  if (ncol(factor_var) != 0) {
    factor_var %>% onehot::onehot() -> onehot_map
    predict(onehot_map,
            jolo %>% as_tibble() %>% select_if(is.factor) )  -> onehot_data
    
    onehot_data %<>% apply(2, as.integer) %>% as_tibble()
    
    jolo %>% as_tibble() %>% select_if(negate(is.factor)) %>%
      bind_cols(onehot_data) -> jolo
  }
}
tokenize_text <- function(zbior,
                          max_len = 100,
                          max_words = 3e4, is_test = F, ...){
  if(is_test == F) {
    tokenizer <<- text_tokenizer(num_words = max_words,
                                 filters = "!\"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n") %>%
      fit_text_tokenizer(zbior$data)
    
    sekwencja <- texts_to_sequences(tokenizer, zbior$data )
    word_index <<- tokenizer$word_index
    cat("##################\n")
    cat("Found", length(word_index), "unique tokens.\n")
    cat("##################\n")
    data <- pad_sequences(sekwencja, maxlen = max_len)
    as_tibble(data) %>%
      mutate(fold = zbior$fold) %>%
      mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
  }else if(is_test == T) {
    sekwencja <- texts_to_sequences(tokenizer = tokenizer, texts = zbior$data)
    data <- pad_sequences(sekwencja, maxlen = max_len)
    as_tibble(data) %>%
      mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
  }
  return(tibb)
}