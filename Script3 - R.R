library(tidyverse)
library(modelr)
library(e1071)
library(lubridate)

#this script was run using an AWS instance
#therefore, the .rds files created in "npz_to_r.R" have been moved to "~/Dropbox/RDS" for access


#Create full dataset
#Spectrometry information is stored in "~/Dropbox/RDS"
files <- list.files("~/Dropbox/RDS")

genres <- read_csv("~/Dropbox/RDS/Genres_songs_fixed.csv")

for (i in seq_along(files)) {
  if (files[i] != "genres.rds" & files[i] != "Genres_songs_fixed.csv") {
    genres <- left_join(genres, read_rds(paste("~/Dropbox/RDS/", files[i], sep = "")))
  }
}

##standardize the spectrometry information
vars <- colnames(genres)[10:71]

for (i in seq_along(vars)) {
  dat <- genres %>%
    pluck(vars[i]) %>%
    unlist
  
  avg <- mean(dat)
  SD <- sd(dat)
  
  tmp <- sym(vars[i])
  
  genres <- genres %>%
    mutate(!!enquo(tmp) := map(!!enquo(tmp), function(x) {
      scale(x, center = avg, scale = SD)[,1]
    }))
}

##save dataset for fast access
write_rds(genres, "~/genres_full_scaled.rds")






#Create reduced size dataset for use with SVM
#(computer crashes when training upon full dataset)

##generate frames per second
frames_per_sec <- genres %>%
  pluck("centroid") %>%
  `[[`(1) %>%
  length() %>%
  `/`(genres %>%
        pluck("Total_Time", 1) %>%
        ms() %>%
        seconds() %>%
        as.numeric())

##split song data into thirds and randomly extract 20 seconds worth of data from each third
one_minute_indexes <- map(genres$centroid, function(x) {
  first <- 1:(floor(length(x) / 3))
  
  second <- if (floor(length(x) / 3) == length(x) / 3) {
    ((length(x) / 3) + 1) : floor(2 * (length(x) / 3))
  } else {
    ceiling(length(x) / 3) : floor(2 * (length(x) / 3))
  }
  
  third <- if (floor(length(x) / 3) == length(x) / 3) {
    ((2 * (length(x) / 3)) + 1) : length(x)
  } else {
    ceiling(2 * (length(x) / 3)) : length(x)
  }
  
  map(list(first, second, third), function(y) {
    if (length(y) > ceiling(frames_per_sec * 20)) {
      sample(y, frames_per_sec * 20) %>%
        sort()
    } else {
      y
    }
  }) %>%
    reduce(c)
})

##reduce data to reduced indexes
for (i in 10:length(genres)) {
  tmp <- map2(pluck(genres, i), one_minute_indexes, function(y, z) {
    y[z]
  })
  
  var <- colnames(genres)[i] %>%
    sym()
  
  genres <- genres %>%
    mutate(!!enquo(var) := tmp)
}

rm(one_minute_indexes, tmp)

##save dataset for fast access
write_rds(genres, "~/genres_reduced_scaled.rds")




#Feature Selection for SVM
rm(list = ls())

genres <- read_rds("~/genres_reduced_scaled.rds")

pures <- genres %>%
  filter(is.na(Genre_2))

rm(genres)

genres_to_omit <- pures %>% 
  group_by(Genre) %>%
  count(Artist) %>%
  select(-n) %>%
  `$`(Genre) %>%
  table() %>%
  `<`(3) %>%
  which() %>%
  names()

pures <- pures %>%
  filter(!(Genre %in% genres_to_omit)) %>%
  mutate(Genre = as_factor(Genre))

#the optimal would be to tune each fitted model
#lack of computing power/time make doing this unreasonable
#interaction variables are a possibility

backward_elim_svm <- function(DATA, KERNEL, FOLDS) {
  fold_list <- caret::createFolds(DATA$Genre, k = FOLDS, list = TRUE)
  
  model_errors <- vector("list", length = 62)
  model_vars <- vector("list", length = 62)
  
  model_vars[[1]] <- 10:71
  
  
  cv_errors <- vector("list", length = FOLDS)
  for (i in 1:FOLDS) { #use majority to test since svm_tune caps out around 6000
    pre_train <- DATA[-fold_list[[i]],] %>%
      unnest()
    
    train <- sample_n(pre_train, 2000)
    while (length(unique(train$Genre)) != length(unique(pre_train$Genre))) {
      train <- sample_n(pre_train, 2000)
    }
    rm(pre_train)
    
    test <- DATA[fold_list[[i]],] %>%
      unnest() %>%
      group_by(Name, Artist, Album, Genre) %>%
      sample_n(100) %>%
      ungroup()
    
    mod <- svm(x = train[10:71], y = train$Genre, kernel = KERNEL, scale = TRUE,
               class.weights = 1/table(train$Genre))
    
    cv_errors[[i]] <- test %>%
      mutate(pred = predict(mod, newdata = test[10:71])) %>%
      group_by(Name, Artist, Album, Genre) %>%
      summarize(preds = list(pred)) %>%
      mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
        x %>%
          table() %>%
          sort(decreasing = TRUE) %>%
          `[`(1) %>%
          names()
      })) %>%
      ungroup() %>%
      summarize(mean(Genre == preds)) %>%
      pluck(1,1)
  }
  
  model_errors[[1]] <- cv_errors %>%
    reduce(c) %>%
    mean()
  
  
  for (j in 1:61) {
    combinations <- combn(model_vars[[j]], length(model_vars[[j]]) - 1, simplify = FALSE)
    
    error <- 0
    
    for (i in seq_along(combinations)) {
      cv_errors <- vector("list", length = FOLDS)
      for (k in 1:FOLDS) {
        pre_train <- DATA[-fold_list[[k]],] %>%
          unnest()
        
        train <- sample_n(pre_train, 2000)
        while (length(unique(train$Genre)) != length(unique(pre_train$Genre))) {
          train <- sample_n(pre_train, 2000)
        }
        rm(pre_train)
        
        test <- DATA[fold_list[[k]],] %>%
          unnest() %>%
          group_by(Name, Artist, Album, Genre) %>%
          sample_n(100) %>%
          ungroup()
        
        mod <- svm(x = train[combinations[[i]]], y = train$Genre, kernel = KERNEL,
                   scale = TRUE, class.weights = 1/table(train$Genre))
        
        cv_errors[[k]] <- test %>%
          mutate(pred = predict(mod, newdata = test[combinations[[i]]])) %>%
          group_by(Name, Artist, Album, Genre) %>%
          summarize(preds = list(pred)) %>%
          mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
            x %>%
              table() %>%
              sort(decreasing = TRUE) %>%
              `[`(1) %>%
              names()
          })) %>%
          ungroup() %>%
          summarize(mean(Genre == preds)) %>%
          pluck(1,1)
      }
      test_error <- cv_errors %>%
        reduce(c) %>%
        mean()
      
      
      if (test_error > error) {
        error <- test_error
        vars <- combinations[[i]]
      }
    }
    
    model_vars[[j + 1]] <- vars
    model_errors[[j + 1]] <- error
    
    if (j > 3) {
      if (error < model_errors[[j]] & model_errors[[j]] < model_errors[[j - 1]] & model_errors[[j - 1]] < model_errors[[j - 2]]) {
        break
      }
    }
  }
  
  reps <- model_errors %>%
    unlist() %>%
    is.na() %>%
    `!`() %>%
    which() %>%
    length()
  
  
  tibble(vars = model_vars[1:reps], error = unlist(model_errors)[1:reps])
}


#Linear SVM
linear_vars <- backward_elim_svm(pures, "linear", 2)

linear_vars %>%
  mutate(vars_removed = map_dbl(vars, function(x) 62 - length(x))) %>%
  ggplot(aes(vars_removed, error)) + geom_point() + geom_line() + xlab("Number of Variables Removed") + ylab("Accuracy (pre-tuned)") + ggtitle("Linear SVM: Backward Elimination") +
  theme_minimal(base_family = "TT Times New Roman")

inds <- caret::createFolds(pures$Genre, k = 2)

#sampled down because the full training set is too large for this computer to handle
train <- pures[inds[[1]],] %>%
  unnest() %>%
  sample_n(3000)
while (length(unique(train$Genre)) != length(unique(pures$Genre))) {
  train <- pures[inds[[1]],] %>%
    unnest() %>%
    sample_n(3000)
}

test <- pures[inds[[2]],] %>%
  unnest()

ideal_vars <- linear_vars$vars %>%
  `[[`(linear_vars$error %>%
         which.max())

#multiple tunes until the ideal cost value is not the minimum or maximum of supplied values
mod <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "linear",
            class.weights = 1/table(train$Genre),
            ranges = list(cost = c(0.1, 1, 10, 50)))
mod2 <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "linear",
             class.weights = 1/table(train$Genre),
             ranges = list(cost = c(50, 75, 100)))
mod3 <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "linear",
             class.weights = 1/table(train$Genre),
             ranges = list(cost = c(100, 150, 200)))
mod4 <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "linear",
             class.weights = 1/table(train$Genre),
             ranges = list(cost = c(200, 250, 300)))

test_pred <- test %>%
  mutate(pred = predict(mod4$best.model, newdata = test[ideal_vars]))

test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  summarize(mean(Genre == preds))
#0.510
#much lower percentage that successful algorithms in literature
#mostly because my database is much more complex and much less far from obvious examples of the genres overlapping than the sets used by academic literature
#also, my training set is only 3000 observations

test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  group_by(Genre) %>%
  summarize(Success_Rate = mean(Genre == preds)) %>%
  mutate(Genre = fct_reorder(Genre, Success_Rate)) %>%
  ggplot(aes(Genre, Success_Rate)) + geom_col() + coord_flip() + ggtitle("Linear SVM") + 
  theme_minimal(base_family = "TT Times New Roman")



#Sample Sizes
test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  group_by(Genre) %>%
  summarize(Success_Rate = mean(Genre == preds)) %>%
  mutate(Genre = fct_reorder(Genre, Success_Rate),
         Sample_Size = map_dbl(Genre, function(x) {
           train %>%
             filter(Genre == x) %>%
             nrow()
         })) %>%
  ggplot(aes(Genre, Sample_Size)) + geom_col() + coord_flip() + ggtitle("Linear SVM") +
  theme_minimal(base_family = "TT Times New Roman")




#Predict Mixed genres
mixed <- read_rds("~/genres_reduced_scaled.rds") %>%
  filter(!is.na(Genre_2))

test_previous <- pures[inds[[2]],]

total <- list(mixed, mutate(test_previous, Genre = as.character(Genre))) %>%
  bind_rows()
rm(test_previous)

total <- total %>%
  filter(!(Genre %in% genres_to_omit))


inds2 <- caret::createFolds(total$Genre, k = 2)

train <- total[inds2[[1]],] %>%
  unnest()
test <- total[inds2[[2]],] %>% 
  unnest()


#assuming absolute (unmixed) frames (i.e. mixed genres have a proportion of genre1 frames and genre2 frames, not a whole of genre1 + genre2 frames)
train_pred <- train %>%
  mutate(pred = predict(mod4$best.model, newdata = train[ideal_vars])) %>%
  group_by(Name, Artist, Album, Genre, Genre_2, Genre_3) %>%
  summarize(preds = list(pred)) %>%
  ungroup()


score <- function(DATA, PARAMETER1, PARAMETER2) {
  mixed_genres <- list(DATA$Genre, DATA$Genre_2, DATA$Genre_3) %>%
    transpose() %>%
    map(unlist)
  
  DATA %>%
    mutate(genres = mixed_genres,
           score = map2_dbl(preds, genres, function(p, g) {
             potentials <- p %>%
               table() %>%
               sort(decreasing = TRUE) %>%
               `[`(1:3) %>%
               '/'(length(p))
             
             if (potentials[3] > PARAMETER1) {
               predictions <- names(potentials)
             } else {
               if (potentials[2] > PARAMETER2) {
                 predictions <- names(potentials[1:2])
               } else {
                 predictions <- names(potentials[1])
               }
             }
             
             x <- g[1]
             y <- g[2]
             z <- g[3]
             length_pred <- length(predictions)
             
             length_actual <- 3 - (c(x, y, z) %>%
                                     is.na() %>%
                                     sum())
             
             n_omit <- c(x, y, z) %>%
               na.omit() %in% 
               predictions %>%
               `!`() %>%
               sum()
             
             n_additional <- predictions %in% na.omit(c(x, y, z)) %>%
               `!`() %>%
               sum()
             
             1 - n_omit*(1 / length_actual) - n_additional*(1 / length_pred)
           }))
}

params <- seq(from = 0, to = 1, length.out = 40) %>%
  combn(m = 2, simplify = FALSE)


basic_scores <- map_dbl(params, function(x) {
  score(train_pred, x[1], x[2]) %>%
    summarize(mean(score)) %>%
    pluck(1,1)
})

ideal_param1 <-  params[[which.max(basic_scores)]][1]
ideal_param2 <-  params[[which.max(basic_scores)]][2]






test_pred <- test %>%
  mutate(pred = predict(mod4$best.model, newdata = test[ideal_vars])) %>%
  group_by(Name, Artist, Album, Genre, Genre_2, Genre_3) %>%
  summarize(preds = list(pred)) %>%
  ungroup() %>%
  score(ideal_param1, ideal_param2)

test_pred %>% 
  summarize(mean(score))
#-0.0481

test_pred %>%
  ggplot(aes(score)) + geom_density() + 
  ggtitle("Linear SVM: Estimated Density Function of Scores") +
  theme_minimal(base_family = "TT Times New Roman")
#Many are semi-correctly classified
#Many are fully in error
#And a much smaller amount are perfectly classified

test_pred %>%
  mutate(status = ifelse(is.na(Genre_2), "Pure", "Mixed")) %>%
  ggplot(aes(score)) + geom_density() + facet_wrap(~status) +
  ggtitle("Linear SVM: Estimated Density Functions of Score") +
  theme_minimal(base_family = "TT Times New Roman")



test_pred <- test_pred %>%
  mutate(prediction = map(preds, function(x) {
    potentials <- x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1:3) %>%
      '/'(length(x))
    
    if (potentials[3] > ideal_param1) {
      predictions <- names(potentials)
    } else {
      if (potentials[2] > ideal_param2) {
        predictions <- names(potentials[1:2])
      } else {
        predictions <- names(potentials[1])
      }
    }
    predictions
  }))

multiple_inds <- test_pred %>%
  `$`(prediction) %>%
  map_dbl(length) %>% 
  `>`(1) %>%
  which()

test_pred[multiple_inds,] %>%
  summarize(mean(score))
#mean score of those which were predicted as multiple: -0.117
test_pred[-multiple_inds,] %>%
  summarize(mean(score))
#mean score of those which were predicted as singular: 0.151
test_pred %>%
  filter(!is.na(Genre_2)) %>%
  summarize(mean(score))
#mean score of those which were actually multiple: -0.323
test_pred %>%
  filter(is.na(Genre_2)) %>%
  summarize(mean(score))
#mean score of those which were actually singular: 0.114

#as expected, pure-genre songs were better classified than multi-genre songs



#provide analysis of which genres were most successful

summary_scores <- map_dbl(present_genres, function(x) {
  test_pred %>%
    filter(Genre == x | Genre_2 == x | Genre_3 == x) %>%
    mutate(success = map_lgl(prediction, function(y) {
      x %in% y
    })) %>%
    summarize(mean(success)) %>%
    pluck(1,1)
})


tibble(Genre = present_genres, successful_detection = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, successful_detection)) %>%
  ggplot(aes(Genre, successful_detection)) + geom_col() + coord_flip() + labs(title = "Linear SVM: Full Test Set", caption = str_wrap('In this plot, "successful_detection" is the average rate of correct classification for each genre. For example, for mixed-genre songs that are partially "Electronic", the model predicted the presence of "Electronic" over 75% of the time.', width = 100)) +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))


#mixed

present_genres <- levels(pures$Genre)

tmp <- test_pred %>%
  filter(!is.na(Genre_2))

#sample is rather small at 658 songs of many genres

summary_scores <- map_dbl(present_genres, function(x) {
  tmp %>%
    filter(Genre == x | Genre_2 == x | Genre_3 == x) %>%
    mutate(success = map_lgl(prediction, function(y) {
      x %in% y
    })) %>%
    summarize(mean(success)) %>%
    pluck(1,1)
})


tibble(Genre = present_genres, successful_detection = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, successful_detection)) %>%
  ggplot(aes(Genre, successful_detection)) + geom_col() + coord_flip() + labs(title = "Linear SVM: Mixed-Genre Songs Only", caption = str_wrap('In this plot, "successful_detection" is the average rate of correct classification for each genre. For example, for mixed-genre songs that are partially "Electronic", the model predicted the presence of "Electronic" over 75% of the time.', width = 100)) +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))


fp_scores <- map_dbl(present_genres, function(x) {
  test_pred %>%
    filter(map_lgl(prediction, function(z) {
      x %in% z
    })) %>%
    mutate(fp = !map_lgl(genres, function(y) {
      x %in% y
    })) %>%
    summarize(sum(fp)) %>%
    pluck(1,1) %>%
    `/`(nrow(test_pred))
})

tibble(Genre = present_genres, false_positive = fp_scores, success = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, success)) %>%
  ggplot(aes(Genre, false_positive)) + geom_col() + coord_flip() + labs(title = "Linear SVM", subtitle = "Final Test Set", caption = "In this plot, 'false_positive' represents the percentage of songs that were fully or partially \nmisclassified as the given genre.") +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))






#Polynomial SVM
genres <- read_rds("~/genres_reduced_scaled.rds")

pures <- genres %>%
  filter(is.na(Genre_2))

rm(genres)

genres_to_omit <- pures %>% 
  group_by(Genre) %>%
  count(Artist) %>%
  select(-n) %>%
  `$`(Genre) %>%
  table() %>%
  `<`(3) %>%
  which() %>%
  names()

pures <- pures %>%
  filter(!(Genre %in% genres_to_omit)) %>%
  mutate(Genre = as_factor(Genre))



poly_vars <- backward_elim_svm(pures, "polynomial", 2)

poly_vars %>%
  mutate(vars_removed = map_dbl(vars, function(x) 62 - length(x))) %>%
  ggplot(aes(vars_removed, error)) + geom_point() + geom_line() + xlab("Number of Variables Removed") + ylab("Accuracy (pre-tuned)") + ggtitle("Polynomial SVM: Backward Elimination") +
  theme_minimal(base_family = "TT Times New Roman")

inds <- caret::createFolds(pures$Genre, k = 2)

train <- pures[inds[[1]],] %>%
  unnest() %>%
  sample_n(3000)
while (length(unique(train$Genre)) != length(unique(pures$Genre))) {
  train <- pures[inds[[1]],] %>%
    unnest() %>%
    sample_n(3000)
}

test <- pures[inds[[2]],] %>%
  unnest()

ideal_vars <- poly_vars$vars %>%
  `[[`(poly_vars$error %>%
         which.max())


mod <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "polynomial",
            class.weights = 1/table(train$Genre),
            ranges = list(cost = c(0.01, 0.05, 0.1, 1),
                          degree = c(2, 3, 4),
                          gamma = c(0.1, 0.5, 1, 2, 3)))


test_pred <- test %>%
  mutate(pred = predict(mod$best.model, newdata = test[ideal_vars]))

test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  summarize(mean(Genre == preds))
#0.500


test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  group_by(Genre) %>%
  summarize(Success_Rate = mean(Genre == preds)) %>%
  mutate(Genre = fct_reorder(Genre, Success_Rate)) %>%
  ggplot(aes(Genre, Success_Rate)) + geom_col() + coord_flip() + ggtitle("Polynomial SVM") + 
  theme_minimal(base_family = "TT Times New Roman")


#Sample Sizes
test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  group_by(Genre) %>%
  summarize(Success_Rate = mean(Genre == preds)) %>%
  mutate(Genre = fct_reorder(Genre, Success_Rate),
         Sample_Size = map_dbl(Genre, function(x) {
           train %>%
             filter(Genre == x) %>%
             nrow()
         })) %>%
  ggplot(aes(Genre, Sample_Size)) + geom_col() + coord_flip() + ggtitle("Polynomial SVM") +
  theme_minimal(base_family = "TT Times New Roman")





#predict mixed genres\
mixed <- read_rds("~/genres_reduced_scaled.rds") %>%
  filter(!is.na(Genre_2))

test_previous <- pures[inds[[2]],]

total <- list(mixed, mutate(test_previous, Genre = as.character(Genre))) %>%
  bind_rows()
rm(test_previous)

for (i in seq_along(genres_to_omit)) {
  total <- total %>%
    filter(Genre != genres_to_omit[i])
}







inds2 <- caret::createFolds(total$Genre, k = 2)

train <- total[inds2[[1]],] %>%
  unnest()
test <- total[inds2[[2]],] %>% 
  unnest()




#assuming absolute frames
train_pred <- train %>%
  mutate(pred = predict(mod$best.model, newdata = train[ideal_vars])) %>%
  group_by(Name, Artist, Album, Genre, Genre_2, Genre_3) %>%
  summarize(preds = list(pred)) %>%
  ungroup()

params <- seq(from = 0, to = 1, length.out = 40) %>%
  combn(m = 2, simplify = FALSE)


basic_scores <- map_dbl(params, function(x) {
  score(train_pred, x[1], x[2]) %>%
    summarize(mean(score)) %>%
    pluck(1,1)
})

ideal_param1 <-  params[[which.max(basic_scores)]][1]
ideal_param2 <-  params[[which.max(basic_scores)]][2]


test_pred <- test %>%
  mutate(pred = predict(mod$best.model, newdata = test[ideal_vars])) %>%
  group_by(Name, Artist, Album, Genre, Genre_2, Genre_3) %>%
  summarize(preds = list(pred)) %>%
  ungroup() %>%
  score(ideal_param1, ideal_param2)

test_pred %>% 
  summarize(mean(score))
#-0.0103

test_pred %>%
  mutate(status = ifelse(is.na(Genre_2), "Pure", "Mixed")) %>%
  ggplot(aes(score)) + geom_density() + facet_wrap(~status) +
  ggtitle("Polynomial SVM: Estimated Density Functions of Score") +
  theme_minimal(base_family = "TT Times New Roman")


test_pred <- test_pred %>%
  mutate(prediction = map(preds, function(x) {
    potentials <- x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1:3) %>%
      '/'(length(x))
    
    if (potentials[3] > ideal_param1) {
      predictions <- names(potentials)
    } else {
      if (potentials[2] > ideal_param2) {
        predictions <- names(potentials[1:2])
      } else {
        predictions <- names(potentials[1])
      }
    }
    predictions
  }))

#provide analysis of which genres were most successful
multiple_inds <- test_pred %>%
  `$`(prediction) %>%
  map_dbl(length) %>% 
  `>`(1) %>%
  which()

test_pred[multiple_inds,] %>%
  summarize(mean(score))
#mean score of those which were predicted as multiple: -0.143
test_pred[-multiple_inds,] %>%
  summarize(mean(score))
#mean score of those which were predicted as singular: 0.389
test_pred %>%
  filter(!is.na(Genre_2)) %>%
  summarize(mean(score))
#mean score of those which were actually multiple: -0.196
test_pred %>%
  filter(is.na(Genre_2)) %>%
  summarize(mean(score))
#mean score of those which were actually singular: 0.0971


summary_scores <- map_dbl(present_genres, function(x) {
  test_pred %>%
    filter(Genre == x | Genre_2 == x | Genre_3 == x) %>%
    mutate(success = map_lgl(prediction, function(y) {
      x %in% y
    })) %>%
    summarize(mean(success)) %>%
    pluck(1,1)
})

tibble(Genre = present_genres, successful_detection = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, successful_detection)) %>%
  ggplot(aes(Genre, successful_detection)) + geom_col() + coord_flip() + labs(title = "Polynomial SVM: Full Test Set", caption = str_wrap('In this plot, "successful_detection" is the average rate of correct classification for each genre. For example, for mixed-genre songs that are partially "Death Metal", the model predicted the presence of "Death Metal" nearly perfectly.', width = 100)) +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))

#mixed
present_genres <- levels(pures$Genre)

tmp <- test_pred %>%
  filter(!is.na(Genre_2))

summary_scores <- map_dbl(present_genres, function(x) {
  tmp %>%
    filter(Genre == x | Genre_2 == x | Genre_3 == x) %>%
    mutate(success = map_lgl(prediction, function(y) {
      x %in% y
    })) %>%
    summarize(mean(success)) %>%
    pluck(1,1)
})

tibble(Genre = present_genres, successful_detection = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, successful_detection)) %>%
  ggplot(aes(Genre, successful_detection)) + geom_col() + coord_flip() + labs(title = "Polynomial SVM: Mixed-Genre Songs Only", caption = str_wrap('In this plot, "successful_detection" is the average rate of correct classification for each genre. For example, for mixed-genre songs that are partially "Death Metal", the model predicted the presence of "Death Metal" nearly perfectly.', width = 100)) +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))

fp_scores <- map_dbl(present_genres, function(x) {
  test_pred %>%
    filter(map_lgl(prediction, function(z) {
      x %in% z
    })) %>%
    mutate(fp = !map_lgl(genres, function(y) {
      x %in% y
    })) %>%
    summarize(sum(fp)) %>%
    pluck(1,1) %>%
    `/`(nrow(test_pred))
})

tibble(Genre = present_genres, false_positive = fp_scores, success = summary_scores) %>%
  `[`(summary_scores %>%
        is.nan() %>%
        `!`() %>%
        which(),) %>%
  mutate(Genre = fct_reorder(Genre, success)) %>%
  ggplot(aes(Genre, false_positive)) + geom_col() + coord_flip() + labs(title = "Polynomial SVM", subtitle = "Final Test Set", caption = "In this plot, 'false_positive' represents the percentage of songs that were fully or partially \nmisclassified as the given genre.") +
  theme_minimal(base_family = "TT Times New Roman") + theme(plot.caption = element_text(hjust = 0.5))


#Radial SVM
genres <- read_rds("~/genres_reduced_scaled.rds")

pures <- genres %>%
  filter(is.na(Genre_2))

rm(genres)

genres_to_omit <- pures %>% 
  group_by(Genre) %>%
  count(Artist) %>%
  select(-n) %>%
  `$`(Genre) %>%
  table() %>%
  `<`(3) %>%
  which() %>%
  names()

pures <- pures %>%
  filter(!(Genre %in% genres_to_omit)) %>%
  mutate(Genre = as_factor(Genre))



rad_vars <- backward_elim_svm(pures, "radial", 2)

rad_vars %>%
  mutate(vars_removed = map_dbl(vars, function(x) 62 - length(x))) %>%
  ggplot(aes(vars_removed, error)) + geom_point() + geom_line() + xlab("Number of Variables Removed") + ylab("Accuracy (pre-tuned)") + ggtitle("Radial SVM: Backward Elimination") +
  theme_minimal(base_family = "TT Times New Roman")

inds <- createFolds(pures$Genre, k = 2)

train <- pures[inds[[1]],] %>%
  unnest() %>%
  sample_n(3000)
while (length(unique(train$Genre)) != length(unique(pures$Genre))) {
  train <- pures[inds[[1]],] %>%
    unnest() %>%
    sample_n(3000)
}

test <- pures[inds[[2]],] %>%
  unnest()

ideal_vars <- rad_vars$vars %>%
  `[[`(rad_vars$error %>%
         which.max())


mod <- tune(svm, train.x = train[ideal_vars], train.y = train$Genre, kernel = "radial",
            class.weights = 1/table(train$Genre),
            ranges = list(cost = c(0.1, 1, 10, 50),
                          gamma = c(0.1, 0.5, 1, 2, 3)))

test_pred <- test %>%
  mutate(pred = predict(mod$best.model, newdata = test[ideal_vars]))

test_pred %>%
  group_by(Name, Artist, Album, Genre) %>%
  summarize(preds = list(pred)) %>%
  mutate(preds = map_chr(preds, function(x) { #returns most frequent prediction
    x %>%
      table() %>%
      sort(decreasing = TRUE) %>%
      `[`(1) %>%
      names()
  })) %>%
  ungroup() %>%
  summarize(mean(Genre == preds))
#0.125
#not  close to polynomial or linear performance
#will not continue to mixed









#Graphical Mapping

genres <- read_rds("~/genres_full_scaled.rds")

frames_per_sec <- genres %>%
  pluck("centroid") %>%
  `[[`(1) %>%
  length() %>%
  `/`(genres %>%
        pluck("Total_Time", 1) %>%
        lubridate::ms() %>%
        lubridate::seconds() %>%
        as.numeric())

five_sec_length <- ceiling(frames_per_sec * 5)


graph_map <- function(DATA, NAME, ARTIST, START_FRAME, VARIABLE = "All", SIZE = "Full") {
  five_sec_length <- 216
  
  ind <- str_c(DATA$Name, DATA$Artist, sep = " - ") %>%
    str_detect(str_c(NAME, ARTIST, sep = " - ")) %>%
    which()
  
  test <- DATA[ind,]
  
  if(SIZE == "Full") {
    train <- DATA[-ind,] %>%
      filter(Artist != test$Artist)
  } else {
    train <- DATA[-ind,] %>%
      filter(Artist != test$Artist) %>%
      sample_n(SIZE)
  }
  
  if (VARIABLE == "All") {
    vars <- colnames(DATA)[10:71]
  } else {
    vars <- VARIABLE
  }
  
  
  if (START_FRAME > (length(unlist(test$centroid)) - 216)) {
    references <- map(vars, function(x) {
      test %>%
        pluck(x) %>%
        unlist() %>%
        `[`((length(unlist(test$centroid)) - 216):length(unlist(test$centroid)))
    })
  } else {
    references <- map(vars, function(x) {
      test %>%
        pluck(x) %>%
        unlist() %>%
        `[`(START_FRAME:(START_FRAME + 216))
    })
  }
  
  results <- map(1:nrow(train), function(y) {
    windows <- map(vars, function(x) {
      train %>%
        pluck(x) %>%
        `[[`(y)
    })
    
    map_dfr(1:(length(windows[[1]]) - 216), function(x) {
      sequence <-  seq(from = x, to = x + 216, by = 1)
      
      errors <- map2_dbl(windows, references, function(win, ref) {
        mean((ref - win[sequence]) ^ 2)
      })
      
      tb <- tibble(Name = train$Name[y], Album = train$Album[y], Artist = train$Artist[y], Genre = train$Genre[y], Genre_2 = train$Genre_2[y], Genre_3 = train$Genre_3[y], Sequence = list(sequence), avg_error = mean(errors))
      
      for (i in seq_along(vars)) {
        var <- sym(vars[i])
        
        tb %>%
          mutate(!!enquo(var) := errors[i])
      }
      
      tb
    }) %>%
      arrange(avg_error) %>%
      `[`(1:20,)
  })
  
  
  results %>%
    bind_rows() %>%
    arrange(avg_error)
}


results <- graph_map(genres, "Carolina", "Kimbra", 5000, VARIABLE = "centroid")


caro <- genres %>%
  filter(Name == "Carolina") %>%
  `$`(centroid) %>%
  unlist() %>%
  `[`(5000:5216)

ref <- genres %>%
  filter(Name == "Run to the Hills") %>%
  `$`(centroid) %>%
  unlist() %>%
  `[`(100:316)

tibble(Name = c("Carolina", "Run to the Hills"), centroid = list(caro, ref), frame = list(1:217)) %>%
  unnest() %>%
  ggplot(aes(frame, centroid, color = Name)) + geom_line() +
  theme_minimal(base_family = "TT Times New Roman")

top_result <- genres %>%
  filter(Name == results$Name[1]) %>%
  `$`(centroid) %>%
  unlist() %>%
  `[`(results$Sequence[[1]])

tibble(Name = c("Carolina", results$Name[1]), centroid = list(caro, top_result), frame = list(1:217)) %>%
  unnest() %>%
  ggplot(aes(frame, centroid, color = Name)) + geom_line() +
  theme_minimal(base_family = "TT Times New Roman")