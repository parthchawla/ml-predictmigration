* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*/
* ------------------------------------------------------------------------------

clear all
macro drop _all
if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
global stats "stats/with_weather"
global output "output/with_weather"

import delimited "$output/lightgbm_feature_importance.csv", clear
gsort - importance


