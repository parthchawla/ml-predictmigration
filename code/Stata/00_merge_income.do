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

use "$data/MexMigData.dta", clear

gen survey = 1 if year <= 2002
replace survey = 2 if year >= 2003 & year < 2008
replace survey = 3 if year >= 2008

merge m:1 numc survey using "$data/totalincome_panel.dta", ///
keepusing(ag_inc asset_inc farmlab_inc liv_inc nonag_inc plot_inc_renta_ag ///
plot_inc_renta_nonag rec_inc rem_mx rem_us trans_inc)
drop if _merge == 2
drop _merge r

save "$data/MexMigData_merged.dta", replace
