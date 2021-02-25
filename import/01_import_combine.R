rm(list = ls())
header <- new.env()
source("/Users/Brian/git/2021_Qiskitties/header.R", local = header)

# Import files and combine them -------------------------------------------

hh_inc  <- read_csv(header$data("median_hh_income2015.csv"))
poverty <- read_csv(header$data("percentage_below_poverty.csv"))
hs      <- read_csv(header$data("percentage_over25_complete_hs.csv"))
race    <- read_csv(header$data("percentage_race_by_city.csv")) %>% 
  rename(`Geographic Area` = `Geographic area`)
deaths  <- read_csv(header$data("police_killings.csv"))

# Census data joins -------------------------------------------------------

census <- full_join(hh_inc, poverty) %>% 
  full_join(hs) %>% 
  full_join(race) %>% 
  clean_names()

write_csv(census, header$int_data("census_joined.csv"))
