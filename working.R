library(tidyverse)
library(data.table)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
library(rebus)
library(keras)
library(furrr)
plan(multisession)
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
train_set %>%
  preprocess() -> zbior


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
                          max_len = 128,
                          max_words = 1024 * 25, is_test = F, use_tf_idf = T,  ...){
  if(is_test == F) {
    if(use_tf_idf == T){
      #--------------------
      zbior %>%
        unnest_tokens(word, data ) %>%
        count(category, word, sort = T) %>%
        bind_tf_idf(word, category, n) %>%
        arrange(desc(tf_idf)) %>%
        distinct(word, .keep_all = T) %>%
        rowid_to_column("value") %>%
        filter(value <= max_words) %>%
        select(value, word) ->> tokenization
      
      to_sequence <<- function(sentence,  ...) {
        sentence %>%
          str_split(" ") %>%
          unlist() %>%
          enframe(name = NULL) %>%
          rename(word = value) %>%
          left_join(tokenization, by= "word") %>%
          pull(value) %>%
          return(.)
      }
      
      cat("Tokenizing with inverse term freq index ... \n")
      zbior$data %>%
        future_map( . , to_sequence , .progress = TRUE) -> sekwencja
      
      tokenization %>%
        split(.$word, lex.order = T) %>%
        map(~.$value) ->> word_index
      
      cat("##################\n")
      cat("Found", length(word_index), "unique tokens.\n")
      cat("##################\n")
      
      data <- pad_sequences(sekwencja, maxlen = max_len)
      as_tibble(data) %>%
        mutate(fold = zbior$fold) %>%
        mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
    } 
    else if(use_tf_idf == F) {
      ####################
      tokenizer <<- text_tokenizer(num_words = max_words,
                                   filters = "!\"#%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n") %>%
        fit_text_tokenizer(zbior$data)
      #--------------------
      sekwencja <- texts_to_sequences(tokenizer, zbior$data)
      word_index <<- tokenizer$word_index
      cat("##################\n")
      cat("Found", length(word_index), "unique tokens.\n")
      cat("##################\n")
      data <- pad_sequences(sekwencja, maxlen = max_len)
      as_tibble(data) %>%
        mutate(fold = zbior$fold) %>%
        mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
    }
  }else if(is_test == T) {
    if(use_tf_idf == T){
      zbior$data %>%
        map( . , to_sequence )  -> sekwencja
      data <- pad_sequences(sekwencja, maxlen = max_len)
      as_tibble(data) %>%
        mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
      tibb[is.na(tibb)] <- 0
    } 
    else if(use_tf_idf == F){
      sekwencja <- texts_to_sequences(tokenizer = tokenizer, texts = zbior$data)
      data <- pad_sequences(sekwencja, maxlen = max_len)
      as_tibble(data) %>%
        mutate(target = as.factor(zbior$category)) %>% onehot_represenation() -> tibb
    }
  }
  return(tibb)
}
#--------------------------------------------
train_set %>%
  preprocess( variable = "main") %>%
  tokenize_text() -> DATA_tokenized
#--------------------------------------------
setwd("..")
lines <- readLines("glove.6B.300d.txt")
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
max_words = 1024 * 25
emdedding_dim <- 300
max_len <- 128
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
train_cross_validate <- function(dane, ... ) {
  progress <- progress_estimated((unique(dane$fold) %>% max()))
  list() -> accuracy_list
  list() -> plot_list
  list() -> history_list
  list() -> model_list
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
    
    #summary(model)
   #deepviz::plot_model(model)
    model %>% compile(
      optimizer = optimizer_adam(),
      loss = "categorical_crossentropy",
      metrics = c("acc")
    )
    ################################
    callback <- list(
      callback_early_stopping(
        monitor = "val_acc",
        min_delta = 0.001,
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
        factor = 0.1,
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
    
    model -> model_list[[f]]
    
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
    plots = plot_list, metrics = history_list, model = model_list
  ))
}
train_cross_validate(dane = DATA_tokenized) -> analisis
write_rds(analisis, "analisis.rds")
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
############################
#tf-idf
############################
library(Rtsne)
library(plotly)
library(ggsci)
show_emb <- function(DATA, variable = "main", top_idf = 500, dims = 2, perplex = 15, ...) {
  if(!top_idf - 1 > 3*perplex) stop("Perplexity too big")
  ggsci::pal_d3() -> paletka
  list() -> lista_emb
  DATA %>%
    preprocess( variable ) %>%
    unnest_tokens(word, data) %>%
    count(category, word, sort = T) %>%
    bind_tf_idf(word, category, n) %>%
    arrange(desc(tf_idf)) -> term_freq_df #%>%
  #group_by(category) %>%
  #top_n(top_idf, tf_idf)
  #-> term_freq_df
  
  select_top_idf <- function(data) {
    data %>%
      top_n(1, tf_idf)
  }
  
  
  term_freq_df %<>%
    nest(-word) %>%
    mutate(prep = map(data, select_top_idf)) %>%
    select(-data) %>%
    unnest() %>%
    group_by(category) %>%
    top_n(top_idf, tf_idf) %>%
    ungroup()
  
  
  find_emb  <- function(term_freq_df, embeddings_index, i,  ...) {
    tibble(word = term_freq_df$word[i]) %>%
      bind_cols(as_tibble(as_tibble(embeddings_index[[term_freq_df$word[i]]]) %>% t())) %>%
      return(.)
  }
  
  map_df(1:nrow(term_freq_df), function(i) find_emb(term_freq_df, embeddings_index, i) ) %>%
    right_join(term_freq_df, by = "word") -> data_after_emb
  data_after_emb %<>% na.omit()
  
  tsne_out <- Rtsne(X = as.matrix(data_after_emb %>% select(starts_with("V"))), dims = dims, perplexity = perplex)
  
  as_tibble(tsne_out$Y) %>%
    bind_cols(data_after_emb %>% select(word, category, tf_idf)) -> tsne_df
  
  tsne_df %>%
    mutate(category = as.factor(category)) %>%
    as_tibble() -> data_to_plot
  
  if(dims == 3){
    data_to_plot %>%
      plot_ly(. , x = ~ V1, y = ~V2, z = ~V3, color = ~ category, text = ~ word, size = ~  tf_idf,  #alpha = 0.8,
              colors = paletka(unique(tsne_df$category) %>% length()),
              hovertemplate = paste(
                "<b>%{text}</b><br><br>",
                "tf_idf: %{marker.size:, }",
                "<extra></extra>"
              )
      ) %>%
      add_markers() %>%
      layout(title = "t-SNE on embedding") -> plocik
  } else if(dims == 2){
    data_to_plot %>%
      plot_ly(. , x = ~ V1, y = ~V2,  color = ~ category, text = ~ word, size = ~  tf_idf,  #alpha = 0.8,
              colors = paletka(unique(tsne_df$category) %>% length()),
              hovertemplate = paste(
                "<b>%{text}</b><br><br>",
                "tf_idf: %{marker.size:, }",
                "<extra></extra>"
              )
      ) %>%
      add_markers() %>%
      layout(title = "t-SNE on embedding") -> plocik
  }
  plocik %>%
    return(.)
  
}
show_emb(DATA) -> tsne



