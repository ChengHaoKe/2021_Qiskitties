rm(list = ls())
header <- new.env()
source("/Users/Brian/git/2021_Qiskitties/header.R", local = header)

# Import files and combine them -------------------------------------------

hh_inc  <- read_csv(header$data("median_hh_income2015.csv"))
poverty <- read_csv(header$data("percentage_below_poverty.csv"))
hs      <- read_csv(header$data("percentage_over25_complete_hs.csv"))
race    <- read_csv(header$data("percentage_race_by_city.csv")) %>% rename(`Geographic Area` = `Geographic area`)
deaths  <- read_csv(header$data("police_killings.csv"))

# Census data joins -------------------------------------------------------

census <- full_join(hh_inc, poverty) %>% 
  full_join(hs) %>% 
  full_join(race) %>% 
  clean_names()

# write_csv(census, header$int_data("census_joined.csv"))

# Get killings locations --------------------------------------------------




# Pull in merged data -----------------------------------------------------

merged <- read_csv(header$int_data("Merged_data.csv")) %>% 
  select(12:25, 2:11) %>% 
  filter(is.na(share_black)) %>% 
  select(City_State) %>% 
  clean_names() %>% 
  distinct(city_state)



# Standardize names -------------------------------------------------------

census2 <- census %>% 
  mutate(city2 = str_replace_all(city, "\\sCDP$", ""),
         city3 = str_replace_all(city2, "\\scity$|\\scity and borough$|\\scity and town$", ""),
         city4 = str_replace_all(city3, "\\stown$", ""),
         city5 = str_replace_all(city4, "\\sborough$|\\sborough.*", ""),
         city6 = str_replace_all(city5, "\\svillage.*", "")) %>% 
  select(-c(city, city2, city3, city4, city5)) %>% 
  rename(city = city6,
         state = geographic_area)

joined <- deaths %>% left_join(census2) %>% 
  filter(is.na(share_black)) %>% 
  distinct(city, state) %>% 
  mutate(place = paste0(city, ", ", state))

# Geocoding ---------------------------------------------------------------

library(ggmap)
locations_df <- mutate_geocode(joined, place)

# write_csv(locations_df, header$int_data("geocoded_deaths.csv"))
# 
# test3  <- geocode(c("Indianapolis", "Santa Barbara"), output = "more")
# test2 <- geocode("Indianapolis", output = "all")

unique_census <- census %>% 
  select(city, geographic_area) %>% 
  mutate(location = paste0(city,", ", geographic_area)) %>% 
  distinct(location)

census_coords <- geocode(unique_census$location, output = "more")

