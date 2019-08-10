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
#--------------------------------------------
train_set %>%
  preprocess( variable = "main") %>%
  tokenize_text() -> DATA_tokenized
#--------------------------------------------
setwd("..")
lines <- readLines("glove.6B.100d.txt")
#-----------------------------------
embeddings_index <- new.env(hash = T,
                            parent = emptyenv() )
cat("Loading embedding... \n")
p <- progress_estimated(length(lines))
###############################
options(expressions = 5e5)
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
  p$tick()$print()
  #if(i %% 1e5 == 0) {gc()}
}
cat("Found", length(embeddings_index), "word vectors.\n")
#-----------------------------------
max_words = 3e4
emdedding_dim <- 100
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
      embedding_matrix[index+1,] <- embedding_vector # not found words will be ALL zero
  }
  q$tick()$print()
}


##################################
#TRAINING
##################################
train_cross_validate <- function(dane, model, callback, ... ) {
  progress <- progress_estimated((unique(dane$fold) %>% max()))
  list() -> accuracy_list
  list() -> plot_list
  list() -> history_list
  for (f in sort(unique(dane$fold)) ) {
    #####################################################
    #MODEL ARCHITECTURE
    #####################################################
    layer_input(shape = list(NULL),
                name = "words") -> layer_words
    
    layer_words %>%
      layer_embedding(input_dim = max_words, output_dim = emdedding_dim, 
                      input_length = max_len, name = "embedding") %>%
      #bidirectional( layer_lstm(units = 128) ) %>%
      bidirectional(layer_gru(units = 64,
                              dropout = 0.1,
                              recurrent_dropout = 0.3,
                              return_sequences = TRUE), name = "bidirectional_gru") %>%
      layer_gru(units = 64, activation = "relu",
                dropout = 0.1,
                recurrent_dropout = 0.3) %>%
      # layer_flatten() %>%
      layer_dense(units = 64 , name = "first_dense_unit", use_bias = F) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      #layer_dropout(0.4) %>%
      layer_dense(units = 32 , name = "2_dense_unit", use_bias = F) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      #layer_dropout(0.1) %>%
      layer_dense(units = 5, activation = "softmax") -> output
    
    model <- keras_model(inputs = layer_words,
                         outputs = output )
    
    get_layer(model, name =  "embedding" ) %>%
      set_weights(list(embedding_matrix)) %>%
      freeze_weights()
    
    summary(model)
    deepviz::plot_model(model)
    model %>% compile(
      optimizer = optimizer_adam(),
      loss = "categorical_crossentropy",
      metrics = c("acc")
    )
    ################################
    callback <- list(
      callback_early_stopping(
        monitor = "val_acc",
        min_delta = 0.01,
        patience = 3,
        restore_best_weights = T
      ),
      # callback_model_checkpoint(
      #   filepath = "my_model.h5",
      #   monitor = "val_acc",
      #   save_best_only = TRUE
      # ),
      callback_reduce_lr_on_plateau(
        monitor = "val_acc",
        factor = 0.5,
        patience = 1 ,
        cooldown = 1,
        verbose = 1
      )
    )
    ###################################
    cat("#################################### \n")
    cat("\n Fold: ", f, "\n")
    #---------------------------------------\
    # f = 1
    # dane %>%
    #   filter(fold == f ) %>%
    #   select(-fold) -> test_data
    
    
    #----------------------------------------
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
    ####################################
    model %>%
      fit(
        verbose = 1,
        x = train_X ,
        y = train_target,
        callbacks = callback,
        validation_data = list(valid_X , valid_target),
        epochs = 30, 
        batch_size = 64 ) -> history #val_acc: 0.9708
    
    history$metrics %>%
      as_tibble() %>% tail(1) %>% 
      select(contains("acc")) %>% 
      select(contains("val")) %>%
      mutate(fold = f) -> accuracy_list[[f]]
    
    plot(history) -> plot_list[[f]]
    history -> history_list[[f]]
    cat("#################################### \n")
    accuracy_list[[f]] %>% print()
    cat("#################################### \n")
    progress$tick()$print()
    k_clear_session()
    # reset_states(model)
  }
  return(list(
    plots = plot_list, metrics = history, model = model
  ))
}
train_cross_validate(DATA_tokenized, model, callback) -> analisis
#################################
test_set %>%
  preprocess( variable = "main") %>%
  tokenize_text(is_test = T) -> test_tokenized

model_performace  <- function(test_tokenized, model) {
  test_tokenized %>%
    select(contains("target")) %>%
    data.matrix() -> test_target
  
  test_tokenized %>%
    select(-one_of(test_target %>% colnames())) %>%
    data.matrix() -> test_X
  
  model %>%
   # load_model_weights_hdf5("my_model.h5") %>%
    evaluate(test_X, test_target) %>%
    as_tibble() %>%
    return(.)
}
model_performace(test_tokenized, model)
##############################################
analisis$plots 

analisis$metrics %>% glimpse()
##############################
#own embedding
################################
library(h2o)
h2o.init()
DATA %>%
  preprocess( variable = "main") %>%
  as.h2o(destination_frame = "DATA") -> DATA_h2o

DATA_h2o$data %>% h2o.ascharacter() %>% h2o.tokenize( . , "\\\\W+") -> token

h2o.word2vec(token, sent_sample_rate = 0, epochs = 10) -> word_2_vec

print(h2o.findSynonyms(word_2_vec, "man", count = 5))
# synonym     score
# 1    hulk 0.5856051
#################################
DATA %>%
  preprocess( variable = "main") %>%
  unnest_tokens(word, data) %>% 
  group_by(category) %>%
  count(word) %>%
  arrange(desc(n)) %>%
  #group_by(category) %>%
  top_n(200, n) %>%
  ungroup() -> top

vector("list", length = nrow(top)) -> embedding_values
#vector("list", length = nrow(top)) -> list_tibble
for (i in 1:nrow(top)) {
  if(is.null( embeddings_index[[top$word[[i]]]])) {
    rep(0,max_len) -> embedding_values[[i]]
  } else {
    embeddings_index[[top$word[[i]]]] -> embedding_values[[i]]
  }
    
    #, rep(0,max_len), embeddings_index[[top$word[[i]]]])
  #embeddings_index[[top$word[[i]]]] -> embedding_values[[i]]
  # tibble(word = top$word[[i]]) %>% cbind(t(
  #   as_tibble(embeddings_index[[top$word[[i]]]])
  # )) -> list_tibble[[i]]
}

#embedding_values[[1]] -> single_embedding
#top[1, ] -> single_name

single_name %>% cbind(single_embedding)
as_tibble(single_name, t(single_embedding) )

top %>% select(word, category) %>% rowid_to_column("id") %>%

embedding_values %>% map(as_tibble) %>%
  map(~prepend(values = as.list(top$word), .x)) %>% reduce(cbind)

embedding_values %>% 
  map(~t(as.data.table(.x))) %>% 
  map(~as.data.table(.x)) %>%
  rbindlist() %>%
  cbind(  word = top$word, category = top$category , .) -> top_embedding
#bind_rows(embedding_values)
embedding_values %>%
  sapply(., function(x) as.data.table(t(as.data.table(x)) )) %>%
  rbindlist() %>%
  as_tibble() %>%
  bind_cols(category = top %>% select(category, word) ,.) %>%
  ungroup() -> top_embedding
###########################################
DATA %>%
  preprocess( variable = "main") %>%
  unnest_tokens(word, data) %>% 
  group_by(category) %>%
  count(word) %>%
  arrange(desc(n)) %>%
  #group_by(category) %>%
  top_n(200, n) %>%
  ungroup() -> top

vector("list", length = nrow(top)) -> embedding_values
for (i in 1:nrow(top)) {
  embeddings_index[[top$word[[i]]]] -> embedding_values[[i]]
  #as_tibble(word = top$word[i]) %>% cbind(data.table(embedding_values[[i]]))
}
#-------------------
embedding_values %>% 
  map(~t(as.data.table(.x))) %>% 
  map(~as.data.table(.x)) %>%
  rbindlist() -> elo

elo$name <- top$word

#--------------------------
embedding_values %>%
  map(. , as.data.table)
  rbindlist()

library(Rtsne)
library(plotly)
as.matrix(top_embedding %>% select(-category, -word)  %>% unique()) %>% duplicated()

top_embedding %>% select(-category, -word)  %>% nrow()
top_embedding %>% select(-category, -word) %>% duplicated() -> index

top_embedding[index, ]$word %>% unique() %>% length()


top_embedding %>%
  nest(-word) %>%
  #group_by(word) %>%
  mutate(data = map(data, ~pull_cat(.x))) %>%
  unnest() -> top_embedding_unique

#-------------

top_embedding %>% select(-category, -word) %>% duplicated() %>% which() %>%
  top_embedding[. , ] %>%
  nest(-word) %>%
  #group_by(word) %>%
  mutate(data = map(data, ~pull_cat(.x))) %>%
  unnest() -> elo
  
 elo$cat_all -> list_categories

elo[ elo$word %>% duplicated() ,]
 #rlist::list.filter()
 elo$word %>% unique() %>% length() 
top_embedding_unique %>%
  identical(nrow(unique(.)), nrow(.))
#--------------

pull_cat <- function(df) {
  df %>% pull(category) -> col_name
  df %>% select(-category) %>% unique() %>%
    mutate(cat_all = list(col_name)) %>% select(cat_all, everything()) %>% return(.)
}
elo$data[[1]] %>% pull(category) %>% toString()

all
tsne_out <- Rtsne(X = as.matrix(top_embedding_unique %>% select(-cat_all, -word)), dims = 3)
top_embedding_unique %>% select(-cat_all, -word) %>% duplicated() %>% which() -> index

top_embedding_unique[index,]$word

# Prepare a data frame for plotting
d_tsne <- data.frame(tsne_out$Y, Class = Vehicle$Class)
colnames(d_tsne) <- c("x", "y", "z", "Class")

# Create a 3D Scatter plot using plotly
p <- 
  plot_ly(d_tsne, x = x, y = y, z = z, type = "scatter3d", group = Class, mode = "markers") %>%
  layout(title = "t-SNE on 'Vehicle'")
