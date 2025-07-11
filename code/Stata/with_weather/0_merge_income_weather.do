* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Nov 24, 2024
*/
* ------------------------------------------------------------------------------

clear all
macro drop _all
if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
global stats "stats"
global output "output"

use "$data/weatherdata.dta", clear
sort village year distancekm
by village year: keep if _n==1 // keep the closest weather station
distinct village year, joint
keep village year RENE-Ttemp_std
tempfile new_weather
save `new_weather'

********************************************************************************
use "$data/MexMigData.dta", clear
gen survey = 1 if year <= 2003 // 2003 survey
replace survey = 2 if year >= 2004 & year <= 2008 // 2008 survey
replace survey = 3 if year > 2008
mdesc survey
keep year numc hhchildren hhworkforce
duplicates drop
distinct year numc, joint
tempfile missing_vars
save `missing_vars'

use "$data/MexMigData.dta", clear
keep ind male
duplicates drop
rename ind indid
tempfile male
save `male'
********************************************************************************

use "$data/LaborWeather_EJ_Main.dta", clear
rename householdid numc

gen survey = 1 if year <= 2003 // 2003 survey
replace survey = 2 if year >= 2004 & year <= 2008 // 2008 survey
replace survey = 3 if year > 2008
mdesc survey

merge m:1 year numc using `missing_vars'
drop if _merge == 2
drop _merge
merge m:1 indid using `male'
drop if _merge == 2
drop _merge

merge m:1 village year using `new_weather'
drop if _merge == 2
drop _merge

merge m:1 numc survey using "$data/totalincome_panel.dta", ///
keepusing(ag_inc asset_inc farmlab_inc liv_inc nonag_inc plot_inc_renta_ag ///
plot_inc_renta_nonag rec_inc rem_mx rem_us trans_inc)
drop if _merge == 2
drop _merge r

save "$data/MexMigData_merged_weather.dta", replace

/*
The data on rural Mexican employment come from the Mexico National Rural Household Survey (Encuesta Nacional a Hogares Rurales de México – ENHRUM), a nationally representative survey of 1,762 households in 80 rural communities spanning Mexico's five census regions. The survey was carried out in the winters of 2003 and 2008.

The 2008 survey asked respondents retrospectively where and in which sector the household head, spouse and all children of either the household head or spouse worked each year beginning in 1990. The household reported whether each family member worked in an agricultural or non-agricultural job and whether the job involved self-employment or wage work. The question was asked for local work, work elsewhere in Mexico and work in the US. For work elsewhere in Mexico, respondents also reported the state in which family members worked. In the 2003 survey, the same format was used to collect employment history retrospective to 1980. One distinction from the 2008 survey is that information was only collected for a randomly chosen subset of individuals in each household. Due to this restriction on the sample, we use the 2008 survey as our primary data set and where possible combine it with the 2003 survey to create a panel of annual data on family members' work histories spanning the period from 1980 to 2007.
*/
