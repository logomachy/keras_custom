library(tidyverse)
library(data.table)
library(tidytext)
library(stringr)
library(tibble)
library(magrittr)
library(rebus)
library(keras)
library(furrr)
library(tfruns)
library(superheat)
library(caret)
library(plotly)
library(uwot)
library(cowplot)
plan(multisession)
#--------------
# library(reticulate)
# 
# library(tensorflow)
#use_condaenv("r-reticulate")
#install_keras(method = "conda", tensorflow = "gpu")

# install_tensorflow(version= "gpu")
# 
# with(tf$device("/gpu:0"), {
#   const <- tf$constant(42)
# })
# 
# reticulate::py_config()
# 
# sess <- tf$Session()
# sess$run(const)

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
# train_set %>%
#   preprocess() -> zbior
# 
#--------------------------------------------------
folds <- 10

preprocess <- function(DATA,
                       variable = "main", no_folds = folds) {
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
tictoc::tic()
for (i in 1:length(lines)) {
  line <- lines[[i]]
  values <- strsplit(line, " ")[[1]]
  word <- values[[1]]
  embeddings_index[[word]] <- as.double(values[-1])
  p$tick()$print()
  #if(i %% 1e5 == 0) {gc()}
}#56.39
tictoc::toc()
###################
names(embeddings_index) %>% tail()
embeddings_index[["mayberg"]]
###################################
# 
# embeddings_index2 <- new.env(hash = T,
#                              parent = emptyenv() )
# load_emb  <- function(i, ...){
#     line <- lines[[i]]
#     values <- strsplit(line, " ")[[1]]
#     word <- values[[1]]
#     #embeddings_index2[[word]] <- as.double(values[-1])
#     data.table(word, values[-1] %>% t()) %>%
#     return(.)
# }
# library(doFuture)
# registerDoFuture()
# mu <- 1.0
# sigma <- 2.0
# x <- foreach(i = 1:3,
#              .export = c("mu", "sigma")) %do%
#   { rnorm(i, mean = mu, sd = sigma) }
# 
# x <- foreach(i = 1:length(lines),
#              .export = c("lines", "embeddings_index2")) %do%
#              { load_emb(i) }
# #######################
# library(future.apply)
# future_sapply(1:length(lines), load_emb,.progress = TRUE) -> elo
# (1:length(lines), load_emb) -> dlo
# 
# foreach (i=1:length(lines),
#          .combine=function(...) rbindlist(list(...)),
#          .multicombine=TRUE) %dopar% load_emb(i) -> elo
# 
# 
# load_emb(212) %>% names()
# 
# lines %>% map(., load_emb)
# 
# lines %>%
#   as_tibble() %>%
#   map(, load_emb)
# 
# lapply(list, function)
# 
# 
# 
# pmap(lines, load_emb) 
# 
# lapply(1:length(lines), function(embeddings_index2, lines, i) pull_lines(embeddings_index2, lines, i) )
# 
# pmap(1:length(lines), pull_lines(embeddings_index2, lines, i))
############################################
cat("Found", length(embeddings_index), "word vectors.\n")
#-----------------------------------
max_words = 1024 * 25
emdedding_dim <- 300
max_len <- 128
#---------------------------------
embedding_matrix <- array(0, c(max_words, emdedding_dim))
q <- progress_estimated(length(names(word_index)))
cat("Filling up the embedding matrix...\n")
tictoc::tic()
for (word in names(word_index)) {
  index <- word_index[[word]]
  if (index < max_words) {
    embedding_vector <- embeddings_index[[word]]
    if (!is.null(embedding_vector))
      embedding_matrix[index+1,] <- embedding_vector # not found words will be ALL zero
  }
  q$tick()$print()
}
tictoc::toc()
#embedding_matrix %>%  write_rds(.,"embedding_matrix.rds")
#read_rds("embedding_matrix.rds") -> embedding_matrix
##################################
#TRAINING
##################################
flagi <- flags(
  flag_integer("first_bidirectional_units", 512),
  flag_numeric("first_bidirectional_drop", 0.1),
  flag_numeric("first_bidirectional_rec_drop", 0.1),


  flag_integer("second_bidirectional_units", 256),
  flag_numeric("second_bidirectional_drop", 0.2),
  flag_numeric("second_bidirectional_rec_drop", 0.2),


  flag_integer("first_dense_units", 128),
  flag_numeric("first_dense_drop", 0.2),

  flag_integer("second_dense_units", 64),
  flag_numeric("second_dense_drop", 0.2)
)


# flagi <- flags(
#   flag_integer("first_bidirectional_units", 64), 
#   flag_numeric("first_bidirectional_drop", 0.1), 
#   flag_numeric("first_bidirectional_rec_drop", 0.1), 
#   
#   
#   flag_integer("second_bidirectional_units", 32), 
#   flag_numeric("second_bidirectional_drop", 0.2), 
#   flag_numeric("second_bidirectional_rec_drop", 0.2),
#   
#   
#   flag_integer("first_dense_units", 32),
#   flag_numeric("first_dense_drop", 0.2),
#   
#   flag_integer("second_dense_units", 16),
#   flag_numeric("second_dense_drop", 0.2)
# )
#################################
reverse_one_hot <- function(tmp) {
  reverse_one_hot_iterator <- function(i, ...) {
    tmp %>% as_tibble() %>%
      .[i, ] %>%
      which.max() %>%
      names() %>%
      return(.)
  }
  sapply(1:nrow(tmp), function(i) reverse_one_hot_iterator(i)) %>%
    str_remove_all("target=") %>%
    enframe(name = NULL) %>%
    mutate(value = as.factor(value)) %>%
    pull() %>%
    return(.)
}
cv_predict <- function(model, valid_X,
                       train_data # for colnames
                       ) {
  model %>% predict(valid_X) %>%
    as_tibble() %>%
    set_names(train_data %>% as_tibble() %>% select(starts_with("target")) %>% colnames()) %>%
    return(.)
}
confusion_matrix_plot <- function(model, valid_X, valid_target, train_data, f) {
  
  cv_predict(model, valid_X, train_data ) -> cross_validation_pred
  confusionMatrix(data=  reverse_one_hot(cross_validation_pred),  reference = reverse_one_hot(valid_target)) -> confuse_a_cat
  confuse_a_cat %>%
    .$byClass %>% as.data.frame() %>% pull(`Balanced Accuracy`) -> Balanced_Accuracy
  confuse_a_cat %>% .$table %>%
    rbind( Balanced_Accuracy) %>%
    superheat(  
      X.text = round(as.matrix(.), 3), order.rows = order( rownames(.) ,decreasing = F),
      row.title = "Prediction",left.label.text.size =  4, bottom.label.text.size  = 4,
      column.title = "Reference", title = paste0("Confusion Matrix fold ", f) ,
      legend = F)
}
train_cross_validate <- function(dane, flagi, epoczki = 10, bacz = 256,  ... ) {
  progress <- progress_estimated((unique(dane$fold) %>% max()))
 # list() -> accuracy_list
  list() -> plot_list
  list() -> history_list #
  list() -> model_list #
  list() -> cross_validation_pred_list #
  list() -> whats_wrong_list #
 # list() -> confusion_matrix_list #
  
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
      bidirectional(layer_gru(units = flagi$first_bidirectional_units,
                              dropout = flagi$first_bidirectional_drop,
                              recurrent_dropout = flagi$first_bidirectional_rec_drop,
                              return_sequences = TRUE), name = paste0("bidirectional_gru_", flagi$first_bidirectional_units)) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_gru(units = flagi$second_bidirectional_units, #activation = "relu",
                dropout = flagi$second_bidirectional_drop,
                recurrent_dropout = flagi$second_bidirectional_rec_drop,
                name = paste0("gru_", flagi$second_bidirectional_units)) %>%
      layer_batch_normalization() %>%
      layer_activation_relu() %>%
      # layer_flatten() %>%
      layer_dense(units = flagi$first_dense_units , name = paste0("first_dense_", flagi$first_dense_units), use_bias = F) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(flagi$first_dense_drop, name = paste0("first_dropout_",flagi$first_dense_drop ) ) %>%
      layer_dense(units = flagi$second_dense_units , name = paste0("second_dense_unit_", flagi$second_dense_units), use_bias = F) %>%
      layer_batch_normalization() %>%
      layer_activation_leaky_relu() %>%
      layer_dropout(flagi$second_dense_drop,  name = paste0("second_dropout_",flagi$second_dense_drop ) ) %>%
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
        patience = 4,
        restore_best_weights = T
      ),  
      callback_model_checkpoint(
        filepath = paste0("model_fold_", f , ".h5"),
        monitor = "val_acc",
        save_best_only = TRUE
      ),
      callback_reduce_lr_on_plateau(
        monitor = "val_acc",
        min_delta = 0.01,
        factor = 0.2,
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
    
    dane[is.na(dane)] <- 0
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
        epochs = epoczki, 
        batch_size = bacz ) -> history_list[[f]]
    ###################
    cv_predict(model, valid_X, train_data )  -> cross_validation_pred_list[[f]] 
    
    tibble(predicted = cross_validation_pred_list[[f]] %>% 
             reverse_one_hot(), 
           true = valid_target %>%
             reverse_one_hot()) %>%
      mutate_all(as.character) %>%
      mutate(zgodny = predicted == true) %>%
      rowid_to_column("id") %>%
      filter(zgodny == F) %>%
      select(-zgodny) -> whats_wrong
    #----------------------
    retokenize_wrong  <- function(wrong_df, whats_wrong, word_index, ...) {
      sapply(word_index, function(x){as.numeric(x[1])}) %>% broom::tidy() -> tidy_word_index
      retokenize_wrong_iterator  <- function(i, ...) {
        wrong_df[i, ] %>%
          t() %>%
          as_tibble() %>%
          left_join(tidy_word_index, by =c("V1" = "x" )) %>%
          pull(names) %>%
          replace_na("0") %>%
          return(.)
      }
      map(1:nrow(wrong_df), function(i) as_tibble(t( retokenize_wrong_iterator(i) %>% str_remove_all("0") %>% .[. != ""] %>% toString() )) ) %>%
        bind_rows() %>%
        bind_cols(whats_wrong , .) %>%
        #rename(word = )
        return(.)
    }
    #---------------------
    valid_X %>% as_tibble() %>%
      .[whats_wrong$id, ] %>%
      retokenize_wrong( . ,whats_wrong, word_index) -> whats_wrong_list[[f]]
    #--------------------------------------------------
    # to do show embedding per id 
    
    # whats_wrong_list[[f]] %>%
    #   unnest_tokens(word, V1 ) -> unnested_wrong
    #   
    # 
    # find_emb  <- function(term_freq_df, embeddings_index, i,  ...) {
    #   tibble(word = term_freq_df$word[i]) %>%
    #     bind_cols(as_tibble(as_tibble(embeddings_index[[term_freq_df$word[i]]]) %>% t())) %>%
    #     return(.)
    # }
    # 
    # map_df(1:nrow(unnested_wrong), function(i) find_emb(unnested_wrong, embeddings_index, i) ) %>%
    #   right_join(unnested_wrong, by = "word")  %>% select(word, id, predicted, true, everything()) -> data_after_emb
    
    #-------------------------------------
    #data_after_emb[is.na(data_after_emb) %>% which(arr.ind = T)]
    
    #data_after_emb  %>% .[is.na(data_after_emb) %>% which(arr.ind = T)]
    
    # data_after_emb[!duplicated(data_after_emb), ] %>%
    #   distinct_if(grep("Sepal|Petal", colnames(.)))
      
      
    # select(-word,-id,-predicted, -true) %>% .[!duplicated(.), ] -> not_duplicated
    #--------------------------------------
    
    #  data_after_emb %>%
    #    distinct_at(. ,vars(paste0("V",1:ncol(data_after_emb %>% select(-word,-id,-predicted,-true)))) ,.keep_all = TRUE) -> good
    #  good[is.na(good)] <- 0
    # # good %>%
    #    
    # 
    #    good %>% select(word,id,predicted,true) %>% 
    #      bind_cols(
    #        as_tibble(
    #          umap(good  %>% select(-word,-id,-predicted,-true) , n_neighbors = 3, learning_rate = 0.05, init = "random", n_components = 2)
    #          )
    #        ) %>%
    #      ggplot(aes(V1,V2)) +
    #       geom_point(aes(color = predicted, shape = true)) -> umap_plot
    #       ggplotly(umap_plot)
    #  
    #       good  %>%
    #         group_by( id) %>%
    #         summarise(text = str_c(word, collapse = " ")) %>%
    #         ungroup()
    #       
    #       tmp$data[[1]] %>% 
    #         
    #         
    #         tidy_austen %>% 
    #         group_by(book, linenumber) %>% 
    #         summarize(text = str_c(word, collapse = " ")) %>%
    #         ungroup()
          
       #-------------------   
          # good %>% select(word,id,predicted,true) %>% 
          #   bind_cols(
          #     as_tibble(
          #       umap(good  %>% select(-word,-id,-predicted,-true) ,y = as.factor(.$true), 
          #            n_neighbors = 15, learning_rate = 0.5, init = "random", n_components = 2)
          #     )
          #   ) %>%
          #   ggplot(aes(V1,V2)) +
          #   geom_point(aes(color = predicted, shape = true)) -> umap_plot2
          # ggplotly(umap_plot2) 
          # 
          # 
          # good %>% select(word,id,predicted,true) %>% 
          #   bind_cols(
          #     as_tibble(
          #       umap(good  %>% select(-word,-id,-predicted,-true) ,y = as.factor(.$true), 
          #            n_neighbors = 15, learning_rate = 0.5, init = "random", n_components = 3)
          #     )
          #   ) %>%
          #   mutate(category = paste0(predicted," | ", true)) %>%
          # plot_ly(data = ., x=~V1, y=~V2, z=~V3, type="scatter3d", mode="markers", size = 1,  color=~category,  text = ~category , 
          #         hovertemplate = paste(
          #           "<b>%{text}</b><br><br>",
          #           #"Number Employed: %{marker.size:,}",
          #           "<extra></extra>"
          #         )
          #         )
          # 
      # mutate(word,id,predicted,true, 
      #            umap(.  %>% select(-word,-id,-predicted,-true), n_neighbors = 5, learning_rate = 0.5, init = "random") )
    
  
    # umap(not_duplicated %>% na.omit(), n_neighbors = 50, learning_rate = 0.5, init = "random") %>%
    #   as_tibble() %>%
    #   bind_cols(
    #     
    #   )
    #--------------------------------------------------
    # CONFUSION MATRIX ACTS AS SIDE EFFECT
    png(paste0("confusion_matrix_fold_", f), height = 900, width = 800)      
    confusion_matrix_plot(model, valid_X, valid_target, train_data, f ) #-> confusion_matrix_list[[f]]
    dev.off()
    #----------------------------------------------
    # save model to h5 file
    paste0("model_fold_", f , ".h5") -> model_list[[f]]
    #----------------------------------------------
    {plot(history_list[[f]]) + 
        ggtitle("Learning History") +
        theme_minimal() +
        theme(legend.position = "none") +
        ggtitle(paste0("Learning on fold ", f))
    } -> plot_list[[f]]
      #ggplotly() -> plot_list[[f]]
    #plot(history) -> plot_list[[f]]
    #history -> history_list[[f]]
    #cat("#################################### \n")
    #accuracy_list[[f]] %>% print()
    #cat("#################################### \n")
    cat("#################################### \n")
    progress$tick()$print()
    k_clear_session()
    # reset_states(model)
  }
  return(list(
    plots = plot_list, history  = history_list, model = model_list,
    cv_prediction = cross_validation_pred_list, wrong = whats_wrong_list
  ))
}
train_cross_validate(dane = DATA_tokenized, flagi) -> analisis
#write_rds(analisis, "analisis.rds")
#read_rds("analisis.rds") -> analisis


#analisis$history[[1]]$metrics$val_acc %>% max()


plot_folds_acc <- function(analisis) {
  analisis$history %>%
    map_dbl(~.$metrics$val_acc %>% max()) %>%
    enframe(name = NULL) %>%
    summarise(mean_val_acc = mean(value),
              sd_val_acc = sd(value)) -> title
  
  title %<>%  apply(. , 2, round, digits = 3) 
  
  p_title <- ggdraw() + 
    draw_label(paste0("val_acc = ",title["mean_val_acc"], " +/- ", title["sd_val_acc"]) , size = 12, fontface = "bold")
  
  analisis$plots %>% cowplot::plot_grid(plotlist = ., p_title) %>%
    return(.)
}

plot_folds_acc(analisis)


#analisis$cv_prediction[[1]] %>% reverse_one_hot()


# analisis$cv_prediction %>%
#   map(~reverse_one_hot(.x))
#analisis$model[[1]]

plot_confusion <- function(folds, png_path = "confusion_matrix_fold_") {
  plot_confusion_iterator <- function(i) {
     cowplot::ggdraw() + cowplot::draw_image(paste0(png_path , i), scale = 1) %>%
      return(.)
  }
  lapply(1:folds, function(i) plot_confusion_iterator(i)) %>%
    return(.)
}
plot_confusion(folds = folds) -> confusion_matrix_list
confusion_matrix_list
# confusion_matrix_list %>%
#   cowplot::plot_grid(plotlist = ., ncol = 2) dont see a thing
#################################
# analisis$metrics[[1]] %>% str()
# analisis$plots[[1]]
# analisis$model[[1]]
#################################
test_set %>%
  preprocess( variable = "main") %>%
  tokenize_text(is_test = T) -> test_tokenized

models_performance_test  <- function(test_tokenized, analisis) {
  model_performace  <- function(test_tokenized, model, i) {
    test_tokenized %>%
      select(contains("target")) %>%
      data.matrix() -> test_target
    
    test_tokenized %>%
      select(-one_of(test_target %>% colnames())) %>%
      data.matrix() -> test_X
  
    load_model_hdf5(model) -> model
    
    
    png(paste0("confusion_matrix_test", i), height = 900, width = 800)      
    confusion_matrix_plot(model, test_X, test_target, test_target, f = paste0("TEST_", i))
    dev.off()
    
    model %>%
      # load_model_weights_hdf5("my_model.h5") %>%
      evaluate(test_X, test_target) %>%
      as_tibble() %>%
      return(.)
  }
  analisis$model %>% length() -> how_many_models
  map_df(1:how_many_models, function(i) model_performace(test_tokenized, analisis$model[[i]], i) ) -> elo
  elo %>%
    select(acc) %>%
    summarise(mean_acc = round(mean(acc), 3),
              sd_acc = round(sd(acc), 3)) -> aggregated_performance
  
  cat("Test set accuracy: ", aggregated_performance$mean_acc , " +/- ", aggregated_performance$sd_acc)
}
models_performance_test(test_tokenized, analisis)
plot_confusion(folds = folds, png_path = "confusion_matrix_test") -> confusion_matrix_list_test
confusion_matrix_list_test
#model_performace(test_tokenized, analisis$model[[1]]) #0.973
#model_performace(test_tokenized, model)


  

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



