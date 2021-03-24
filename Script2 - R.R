library(tidyverse)
library(reticulate)

np <- import("numpy")


add_feature <- function(file_key, variable_name) {
  tmp <- map(0:20, function(x) {
    path <-  str_c("/Volumes/Benjamin's iTunes Library/Music_Analysis/", file_key, "/batch", x, ".npz", sep = "")
    
    npz <- np$load(path)
    locations <- npz$f[["kwds"]]
    
    names_arrays <- npz$files[-1]
    
    values <- map(names_arrays, function(z) npz$f[[z]])
    
    num_var <- dim(values[[1]])[1]
    
    if (is.matrix(values[[1]])) {
      data_list <- map(1:num_var, function(z) {
        map(1:length(values), function(y) {
          values[[y]][z,]
        })
      })
      
      df <- tibble(Location = locations)
      for (i in 1:num_var) {
        variable <- paste(variable_name, i, sep = "")
        var = sym(variable)
        
        df <- df %>%
          mutate(!!enquo(var) := data_list[[i]])
      }
      
      df
    } else {
      variable = sym(variable_name)
      
      tibble(Location = locations, !!enquo(variable) := values)
    }
  }) %>%
    bind_rows()
  
  tmp[!duplicated(tmp["Location"]),]
}



features <- c("centroid", "chromagram", "mfcc", "polys_order0", "polys_order1", "polys_order2", "polys_order3", "rms", "rolloff10", "rolloff85",
              "spectral_bandwidth", "spectral_contrast", "spectral_flatness", "tonal_centroid_features", "zero_crossing_rates")

for (i in seq_along(features)) {
  add_feature(features[i], features[i]) %>%
    write_rds(str_c("/Volumes/Benjamin's iTunes Library/Music_Analysis/", features[i], ".rds", sep = ""))
}

