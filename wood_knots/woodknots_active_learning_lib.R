entropy <- function(p1, p2){
  H <- function(p){
    p[0>p] <- NA
    p[0==p] <- 1e-100
    -p * log2(p)
  }
  p3 <- 1.0 - p1 - p2
  p3 <- ifelse(p3 < -1e-6, NA, p3)
  p3 <- ifelse(p3 < 0, 0, p3)
  H(p1) + H(p2) + H(p3)
}

get_pseudolabelling_function <- function(plabels_file, knot_classes=KNOT_CLASSES){
  pseudolabels <- local({
    pl <- read.csv(plabels_file, row.names="knot_file", stringsAsFactors=FALSE)
    setNames(pl$knot_class, row.names(pl))
  })
  
  function(file_info){
    file_info %>%
      #mutate(knot_class = factor(pseudolabels[path], levels=knot_classes))
      mutate(knot_class = pseudolabels[path])
  }
}



get_new_pseudolabelled_sample <- function(labelled_filenames, unlabelled_knot_data_df){
  labelled_filenames <- labelled_filenames[!is.na(labelled_filenames$path),]
  row.names(labelled_filenames) <- labelled_filenames$path
  pls <- unlabelled_knot_data_df[unlabelled_knot_data_df$path %in% labelled_filenames$path, ]
  pls$knot_class <- labelled_filenames[pls$path, "knot_class"]
  pls
}


fit_and_evaluate_model <- function(candidate_cases, form=FORM, test_set=TEST_SET){
  
  compute_roc <- function(dframe){
    library(pROC)
    dframe$is_sound <- dframe$knot_class == "sound_knot"
    dframe$is_encased <- dframe$knot_class == "encased_knot"
    dframe$is_dry <- dframe$knot_class == "dry_knot"
    list(sound = roc(is_sound ~ sound_knot, dframe, direction="<"),
         dry = roc(is_dry ~ dry_knot, dframe, direction="<"),
         encased = roc(is_encased ~ encased_knot, dframe, direction="<"))
  }
  
  prediction_entropy <- function(dframe){
    with(dframe, entropy(sound_knot, dry_knot))
  }
  
  prediction_logloss <- function(dframe){
    log_probs <- sapply(1:nrow(dframe), function(i) {
      pclass <- as.character(dframe$pred_class[i])
      log(dframe[i, pclass])
    })
    -sum(log_probs)/nrow(dframe)
  }
  
  # NOTE: candidate_cases may include cases that are not of the classes we are modeling. 
  # We depend on labellers to remove these
  training_set_new <- candidate_cases %>% filter(knot_class %in% KNOT_CLASSES)
  #training_set_new$knot_class <- factor(as.character(training_set_new$knot_class), levels=KNOT_CLASSES)
  

  sken <- import("sklearn.ensemble")
  model <- sken$RandomForestClassifier(n_estimators=101L, n_jobs=-1L, max_depth=4L)
  
    
  X_train <- model.matrix(form, training_set_new)
  model$fit(X=X_train, y=training_set_new[['knot_class']])
  
  X_test <- model.matrix(form, test_set)
  
  pred_prob <- model$predict_proba(X_test)
  pred_prob <- as.data.frame(pred_prob)
  names(pred_prob) <- model$classes_
  
  pred_class <- model$classes_[apply(pred_prob, 1, which.max)]
  
  pred_test <- data.frame(path = test_set$path,
                          knot_class = test_set$knot_class,
                          pred_class = pred_class)
  
  pred_test <- cbind(pred_test, pred_prob)
  
  roc_list <- compute_roc(pred_test)
  
  results <- list(
    model=model,
    tss=nrow(training_set_new),
    test_predictions=pred_test,
    # selected=selected, # unlabelled
    roc_list = roc_list,
    performance = c(accuracy=with(pred_test, sum(pred_class == knot_class)/length(knot_class)),
                    neg_logloss = -prediction_logloss(pred_test),
                    auc_sound = auc(roc_list[['sound']]),
                    auc_dry = auc(roc_list[['dry']]),
                    auc_encased = auc(roc_list[['encased']]),
                    negentropy = -mean(prediction_entropy(pred_test))
    ),
    confusion = with(pred_test, table(knot_class, pred_class))
  )
  
  return(results)
}


select_cases <- function(model, available_cases, N=ADDITIONAL_CASES_TO_LABEL){
  
  test_data <- available_cases
  test_data <- model.matrix(FORM, test_data)
  pred_prob <- model$predict_proba(test_data)
  pred_prob <- as.data.frame(pred_prob)
  names(pred_prob) <- model$classes_
  
  pred_class <- model$classes_[apply(pred_prob, 1, which.max)]
  
  predictions <- data.frame(path = available_cases$path,
                          pred_class = pred_class)
  
  predictions_df <- cbind(predictions, pred_prob)
  
  selected <- predictions_df %>% 
    mutate(entropy=entropy(encased_knot, dry_knot)) %>% 
    top_n(N, entropy) %>% 
    as.data.frame
  
  return(selected)
  
} 

plot_roc_history <- function(kclass, initial_results, results_history){
  roc0 <- initial_results$roc_list[[kclass]]
  plot(roc0, main=kclass, col="black")
  text(0.15, 0, sprintf("AUC: %0.3f", auc(roc0)), col="black", pos=3, cex=0.8)
  rbow <- rainbow(length(results_history), end=2/3, v=0.75)
  for (i in seq_along(results_history)){
    roc <- results_history[[i]]$roc_list[[kclass]]
    lines(roc, col=rbow[i])
    text(0.15, i * 0.04, sprintf("AUC: %0.3f", auc(roc)), col=rbow[i], pos=3, cex=0.8)
  }
}

plot_class_separation <- function(predictions, previous=NULL, ...){
  
  trail <- function(x, y, crp=colorRampPalette(c("white", "red")), num_steps=10, ...){
    colors <- crp(num_steps)
    for(k in 2:length(x)){
      xx <- seq(x[k-1], x[k], len=num_steps+1)
      yy <- seq(y[k-1], y[k], len=num_steps+1)
      points(xx[1:num_steps], yy[1:num_steps], col=colors, cex=0.5, ...)
    }
  }
  
  x <- y <- seq(0,1, len=300)
  z <- outer(x, y, entropy)
  image(x, y, z, asp=1, 
        xlab="sound_knot", ylab="dry_knot",
        col=colorRampPalette(c("yellow", "white"))(16),
        xlim=c(-0.05, 1.05), ylim=c(-0.05, 1.05),
        ...)
  
  point_colors <- c("red", "darkgreen","blue")
  point_symbols <- c(1, 2, 4)
  with(predictions, {
    points(sound_knot, dry_knot, 
           col=point_colors[knot_class], 
           pch=point_symbols[knot_class])
    legend("topright", legend=levels(knot_class), 
           text.col=point_colors, col=point_colors, 
           pch=point_symbols, bty="n")
    
    if (!is.null(previous)){
      for (j in 1:nrow(previous)){
        trail(c(previous$sound_knot[j], sound_knot[j]), c(previous$dry_knot[j], dry_knot[j]), 
              crp=colorRampPalette(c("white", point_colors[knot_class[j]])), pch=point_symbols[knot_class[j]])
      }
      
    }
    
  })
}


plot_class_histograms <- function(pred){
  pred %>% 
    gather(category, score, -c(path, knot_class, pred_class)) %>%
    mutate(category = factor(paste0(category, "_score"), levels=paste0(levels(pred$knot_class), "_score"))) %>%
    ggplot(aes(x=score, fill=knot_class)) + geom_histogram(bins=50) + facet_grid( category ~ .)
}

