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

use "$data/daily_weather_all.dta", clear
distinct village year, joint
tempfile daily_weather
save `daily_weather'

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

merge m:1 village year using `daily_weather'
drop if _merge == 2
drop _merge

merge m:1 numc survey using "$data/totalincome_panel.dta", ///
keepusing(ag_inc asset_inc farmlab_inc liv_inc nonag_inc plot_inc_renta_ag ///
plot_inc_renta_nonag rec_inc rem_mx rem_us trans_inc)
drop if _merge == 2
drop _merge r

save "$data/MexMigData_daily_weather.dta", replace
