# Constant paths --------------------------------------------------------
DATAROOT <- '~/git/2021_Qiskitties/raw_data'

# Options ---------------------------------------------------------------
options(scipen = 999)

# Path functions --------------------------------------------------------

path_builder <- function(path = '', root_func = NULL) { 
  function(file_path = '') {
    if (is.null(root_func)) {
      file.path(path, file_path)
    } else {
      root_func(file.path(path, file_path))
    }
  }
}

data <- path_builder(DATAROOT)
import    <- path_builder('import',   data_root)
build     <- path_builder('build',    data_root)
datasets  <- path_builder('datasets', data_root)
export    <- path_builder('export',   data_root)
analysis  <- path_builder('analysis', data_root)

# Libraries -------------------------------------------------------------
library(readr)
library(janitor)
library(glue)
library(purrr)
library(stringr)
library(fs)
library(dplyr)
library(openxlsx)
library(lubridate)
library(tidyr)
library(ggplot2)
library(zoo)
library(data.table)
library(broom)
library(readxl)
# library(httr)
# library(jsonlite)
# library(usmap)
# library(ggmap)
# library(sf)
# library(sp)
# library(maps)
# library(maptools)
# library(caret)
library(leaps)
library(bestglm)
library(glmnet)
# library(tigris)
library(car)
# library(zipcodeR)
library(tidyverse)
# library(tmap)
library(survival)
library(rms)
library(survminer)
library(expm)

calc_rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

