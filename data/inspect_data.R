library(readr)
library(tidyverse)

combined_data <- read_csv("data/dataset.csv")
data_filtered <- combined_data %>% 
  select(respondent = SEQN,
    gender = RIAGENDR,  # DEMO_L
    age_years = RIDAGEYR,  
    age_months = RIDAGEMN,  
    race_ethnicity = RIDRETH1,  
    race_ethnicity_nh_asian = RIDRETH3,  
    exam_time_period = RIDEXMON,  
    exam_age_months = RIDEXAGM,  
    military_service = DMQMILIZ,  
    income_poverty_ratio = INDFMPIR,  
    birth_country = DMDBORN4,  
    years_in_us = DMDYRUSR,  
    education_level_adult = DMDEDUC2,  
    marital_status = DMDMARTZ,  
    pregnancy_status = RIDEXPRG,  
    household_size = DMDHHSIZ,  
    hh_ref_gender = DMDHRGND,  
    hh_ref_age_years = DMDHRAGZ,  
    hh_ref_education = DMDHREDZ,  
    hh_ref_marital_status = DMDHRMAZ,  
    hh_spouse_education = DMDHSEDZ,
    alcohol_before_lab = PHQ030, # FASTQX_L
    gum_before_lab = PHQ040,
    diet_supplements_before_lab = PHQ060,
    ever_breastfed_before_lab = DBQ010, 
    total_cholesterol_mg_dL = LBXTC, # TCHOL_L
    source_of_food = DR1FS, # DR1IFF_L
    energy = DR1IKCAL,
    protein = DR1IPROT,
    carbohydrate = DR1ICARB,
    total_sugars = DR1ISUGR,
    dietary_fiber = DR1IFIBE,
    total_fat = DR1ITFAT,
    saturated_fat = DR1ISFAT,
    monounsaturated_fat = DR1IMFAT,
    polyunsaturated_fat = DR1IPFAT,
    cholesterol = DR1ICHOL,
    vitamin_e_alpha_tocopherol = DR1IATOC,
    added_alpha_tocopherol = DR1IATOA,
    retinol = DR1IRET,
    vitamin_a_rae = DR1IVARA,
    alpha_carotene = DR1IACAR,
    beta_carotene = DR1IBCAR,
    beta_cryptoxanthin = DR1ICRYP,
    lycopene = DR1ILYCO,
    lutein_zeaxanthin = DR1ILZ,
    thiamin = DR1IVB1,
    riboflavin = DR1IVB2,
    niacin = DR1INIAC,
    vitamin_b6 = DR1IVB6,
    total_folate = DR1IFOLA,
    folic_acid = DR1IFA,
    food_folate = DR1IFF,
    folate_dfe = DR1IFDFE,
    total_choline = DR1ICHL,
    vitamin_b12 = DR1IVB12,
    added_vitamin_b12 = DR1IB12A,
    vitamin_c = DR1IVC,
    vitamin_d = DR1IVD,
    vitamin_k = DR1IVK,
    calcium = DR1ICALC,
    phosphorus = DR1IPHOS,
    magnesium = DR1IMAGN,
    iron = DR1IIRON,
    zinc = DR1IZINC,
    copper = DR1ICOPP,
    sodium = DR1ISODI,
    potassium = DR1IPOTA,
    selenium = DR1ISELE,
    caffeine = DR1ICAFF,
    theobromine = DR1ITHEO,
    alcohol = DR1IALCO,
    ever_had_alcohol = ALQ111, # ALQ_L
    freq_drink_past_12mos = ALQ121,
    avg_drinks_per_day_past_12mos = ALQ130,
    days_4_5_drinks_past_12mos = ALQ142,
    times_4_5_drinks_in_2hrs_past_12mos = ALQ270,
    times_8plus_drinks_in_1day_past_12mos = ALQ280,
    ever_daily_4_5_drinks = ALQ151,
    times_4_5_drinks_past_month = ALQ170, 
    ever_had_high_bp = BPQ020,  # BPQ_L
    told_high_bp_2plus_times = BPQ030,  
    taking_bp_medication = BPQ150,  
    told_high_cholesterol = BPQ080,  
    taking_cholesterol_meds = BPQ101D,
    family_poverty_category = INDFMMPC, # INQ_L
    little_interest = DPQ010,  # DPQ_L
    feeling_down = DPQ020,  
    sleep_trouble = DPQ030,  
    low_energy = DPQ040,  
    poor_appetite_or_overeating = DPQ050,  
    feeling_bad_about_self = DPQ060,  
    trouble_concentrating = DPQ070,  
    moving_or_speaking_irregular = DPQ080,  
    thoughts_of_self_harm = DPQ090,
    moderate_ltpa_freq = PAD790Q,  # PAQ_L
    moderate_ltpa_unit = PAD790U,  
    moderate_ltpa_minutes = PAD800,  
    vigorous_ltpa_freq = PAD810Q,  
    vigorous_ltpa_unit = PAD810U,  
    vigorous_ltpa_minutes = PAD820,  
    sedentary_minutes = PAD680,
    weekday_sleep_time = SLQ300,  # SLQ_L
    weekday_wake_time = SLQ310,  
    weekday_sleep_hours = SLD012,  
    weekend_sleep_time = SLQ320,  
    weekend_wake_time = SLQ330,  
    weekend_sleep_hours = SLD013) %>%
    distinct()

data_filtered <- data_filtered %>%
  select(where(~ sum(is.na(.)) < 6000))

data_averaged <- data_filtered %>%
  group_by(respondent) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE)) %>%
  distinct()

data_averaged <- data_averaged %>%
  filter(!is.na(energy))


data_clean <- data_averaged %>%
  mutate(gender = recode(gender, `1` = "Male", `2` = "Female"),
         race_ethnicity = recode(race_ethnicity, 
                          `1` = "Mexican", 
                          `2` = "Other Hispanic", 
                          `3` = "NH White", 
                          `4` = "NH Black", 
                          `5` = "Other"),
         race_ethnicity_nh_asian = recode(race_ethnicity_nh_asian, 
                                          `1` = "Mexican", 
                                          `2` = "Other Hispanic", 
                                          `3` = "NH White", 
                                          `4` = "NH Black", 
                                          `6` = "NH Asian",
                                          `7` = "Other"),
         exam_time_period = recode(exam_time_period, 
                            `1` = "November 1 through April 30", 
                            `2` = "May 1 through October 31"),
         birth_country = recode(birth_country, 
                         `1` = "Born in 50 US states or Washington", 
                         `2` = "Others",
                         `77` = "Refused",
                         `99` = "Don't know",),
         alcohol_before_lab = recode(alcohol_before_lab, `1` = "Yes", `2` = "No"),
         gum_before_lab = recode(gum_before_lab, `1` = "Yes", `2` = "No"),
         diet_supplements_before_lab = recode(diet_supplements_before_lab, `1` = "Yes", `2` = "No"))


write_csv(data_clean, "data/clean_data.csv")
