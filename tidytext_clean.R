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
#caret::createDataPartition(DATA$category, p = 0.8, list = F) -> in_train 
#DATA %<>% mutate(in_train = ifelse(name %in% in_train, T, F))
#DATA %<>% mutate(text = map_chr(text, unlist )) 
###############
DATA %>%
  group_by(category) %>%
  sample_n(1) -> sample_data
##############
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
    nest(-id, -category) %>% 
    mutate(data = map_chr( data, collapse_word) ) %>%
    mutate(fold = folds) %>%
    return(.)
}
tokenize_text <- function(zbior, 
                          max_len = 100,
                          max_words = 1e4){
  tokenizer <- text_tokenizer(num_words = max_words,
                              filters = "!\"#%&()$*+,-./:;<=>?@[\\]^_`{|}~\t\n") %>%
    fit_text_tokenizer(zbior$data)
  
  sekwencja <- texts_to_sequences(tokenizer, zbior$data )
  word_index <<- tokenizer$word_index
  cat("##################\n")
  cat("Found", length(word_index), "unique tokens.\n")
  cat("##################\n")
  data <- pad_sequences(sekwencja, maxlen = max_len)
  #----------------------
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
  #-----------------------
  as.tibble(data) %>%
    mutate(fold = zbior$fold) %>%
    mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
  #--------------------__
  #cat()
  return(list(DATA = tibb, WORD_INDEX =  word_index))
}
#############
DATA %>% 
  preprocess( variable = "main") %>%
  tokenize_text() -> elo
#------------
DATA %>% 
  preprocess( variable = "header") -> tmp

tmp -> zbior
####################
# LOADING EMB
###################
#list.files(getwd() %>% str_sub(end = -4))
setwd("..")
lines <- readLines("glove.6B.300d.txt")
#-----------------------------------
embeddings_index<- new.env(hash = T,
                           parent = emptyenv() )
cat("Loading embedding... \n")
p <- progress_estimated(length(lines))
options(expressions = 5e5)
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
  p$tick()$print()
}
cat("Found", length(embeddings_index), "word vectors.\n")
########################################################################
#----------------------------------
max_words = 1e4
emdedding_dim <- 300
max_len <- 100
#---------------------------------
embedding_matrix <- array(0, c(max_words, emdedding_dim))
q <- progress_estimated(length(names(word_index)))
cat("Filling up the embedding matrix...\n")
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector
  }
  q$tick()$print()
}
#-------------------------------------
#######################################################################

######################################################################
layer_input(shape = list(NULL),
            name = "words") -> layer_words

layer_words %>%
  layer_embedding(input_dim = max_words, output_dim = emdedding_dim, 
                  input_length = max_len, name = "embedding") %>%
  bidirectional( layer_lstm(units = 128, dropout = 0.4, recurrent_dropout = 0.3) ) %>%
  #layer_dense(units = 128 , name = "first_dense_unit", use_bias = F) %>%
  #layer_batch_normalization() %>%
  #layer_activation_leaky_relu() %>%
  #layer_dropout(0.4) %>%
  layer_dense(units = 64 , name = "first_dense_unit", use_bias = F) %>%
  layer_batch_normalization() %>%
  layer_activation_leaky_relu() %>%
  layer_dropout(0.4) -> layer_base

layer_business <- layer_base %>%
  layer_dense(units = 1, activation = "sigmoid", name = "business")

layer_entertainment <- layer_base %>%
  layer_dense(units = 1, activation = "sigmoid", name = "entertainment")

layer_politics <- layer_base %>%
  layer_dense(units = 1, activation = "sigmoid", name = "politics")

layer_sport <- layer_base %>%
  layer_dense(units = 1, activation = "sigmoid", name = "sport")

layer_tech <- layer_base %>%
  layer_dense(units = 1, activation = "sigmoid", name = "tech")

model <- keras_model(inputs = layer_words,
                     outputs = list(layer_business, layer_entertainment, layer_politics,
                                    layer_sport, layer_tech))

get_layer(model, name =  "embedding" ) %>%
  set_weights(list(embedding_matrix)) %>%
  freeze_weights()

summary(model)

model %>% compile(
  optimizer = optimizer_adam(),
  loss =  list(
    business = "binary_crossentropy",
    entertainment = "binary_crossentropy",
    politics = "binary_crossentropy",
    sport = "binary_crossentropy",
    tech = "binary_crossentropy"
  ), 
  metrics = c("acc")
)

elo[[1]] -> x_train
elo[[2]] -> y_train



elo$DATA -> dane
progress <- progress_estimated((unique(dane$fold) %>% max()))
list() -> accuracy_list
list() -> plot_list
for (f in sort(unique(dane$fold)) ) {
  cat("#################################### \n")
  cat("\n Fold: ", f, "\n")
  
  dane %>%
    filter(fold != f ) %>%
    select(-fold) -> train_data
  
  dane %>%
    filter(fold == f ) %>%
    select(-fold) -> valid_data
  
  train_data %>%
    select(contains("target")) %>%
    data.matrix() -> train_target
  
  train_data %>%
    select(-one_of(train_target %>% colnames())) %>%
    data.matrix() -> train_X

  valid_data %>%
    select(contains("target")) %>%
    data.matrix() -> valid_target
  
  valid_data %>%
    select(-one_of(train_target %>% colnames())) %>%
    data.matrix() -> valid_X

  model %>%
    fit(
      verbose = 1,
      x = train_X ,
      y = list(train_target[,1], train_target[,2], train_target[,3], 
               train_target[,4], train_target[,5]),
      validation_data = list(valid_X ,
                             list(valid_target[,1], valid_target[,2], valid_target[,3],
                                  valid_target[,4], valid_target[,5])),
      epochs = 10, 
      batch_size = 32 ) -> history
  
  history$metrics %>%
    as.tibble() %>% tail(1) %>% 
    select(contains("acc")) %>% 
    select(contains("val")) %>%
    mutate(fold = f) -> accuracy_list[[f]]
  
  plot(history) -> plot_list[[f]]
  cat("#################################### \n")
  accuracy_list[[f]] %>% print()
  cat("#################################### \n")
  progress$tick()$print()
}




plot(history)


#################
as.integer(as.factor(zbior$category)) -> target
{target - 1 } %>% to_categorical(num_classes = unique(DATA$category) %>% length()) -> target



