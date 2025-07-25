* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Sep 30, 2024
*/
* ------------------------------------------------------------------------------
cls
clear all
macro drop _all
if inlist("`c(username)'","parthchawla1") global path ///
"/Users/parthchawla1/GitHub/ml-predictmigration"
else global path ""
cd "$path"
global data "data"
global stats "stats"
global output "output"

import delimited "$output/test_predictions_2010_add_vars1.csv", clear
gen correct_prediction = (actual_y==predicted_y)
tab correct_prediction

////////////////////////////////////////////////////////////////////////////////
* Migrants
////////////////////////////////////////////////////////////////////////////////

tab work_us
* 344 migrants in 2010

gen correct_migrant = (correct_prediction==1 & work_us==1)
tab correct_migrant
* 335/344 migrants correctly predicted, 9 incorrect

gen stayed_migrant = (work_us==1 & l1_work_us==1)
tab stayed_migrant
* 328 were migrants in 2009 and stayed in 2010

gen new_migrant = (work_us==1 & l1_work_us==0)
tab new_migrant
* 16 changed status from nonmigrant to migrant in 2010

gen new2_migrant = (work_us==1 & l1_work_us==0 & l2_work_us==0)
tab new2_migrant
* 14 were migrants in 2010 but not 2009 or 2008

gen correct_stayed_migrant = (correct_migrant==1 & l1_work_us==1)
tab correct_stayed_migrant
* 328/328 of those who stayed migrant correctly predicted, 0 incorrect

gen correct_new_migrant = (correct_migrant==1 & l1_work_us==0)
tab correct_new_migrant
* 7/16 of new migrants correctly predicted, 9 incorrect

gen correct_new2_migrant = (correct_migrant==1 & l1_work_us==0 & l2_work_us==0)
tab correct_new2_migrant
* 5/14 of new2 migrants correctly predicted, 9 incorrect

gen seminew_migrant = (work_us==1 & l1_work_us==1 & l2_work_us==0)
tab seminew_migrant
* 8 were migrants in 2010 and 2009 but not 2008

gen correct_seminew_migrant = (correct_migrant==1 & l1_work_us==1 & l2_work_us==0)
tab correct_seminew_migrant
* 8/8 seminew migrants correctly predicted, 0 incorrect

////////////////////////////////////////////////////////////////////////////////
* Non-migrants
////////////////////////////////////////////////////////////////////////////////

tab work_us
* 5,645 non-migrants in 2010

gen correct_nonmigrant = (correct_prediction==1 & work_us==0)
tab correct_nonmigrant
* 5,518/5,645 nonmigrants correctly predicted, 127 incorrect

gen stayed_nonmigrant = (work_us==0 & l1_work_us==0)
tab stayed_nonmigrant
* 5,616 were nonmigrants in 2009 and stayed in 2010

gen new_nonmigrant = (work_us==0 & l1_work_us==1)
tab new_nonmigrant
* 29 changed status and stopped migrating in 2010

gen correct_stayed_nonmigrant = (correct_nonmigrant==1 & l1_work_us==0)
tab correct_stayed_nonmigrant
* 5,518/5,616 of those who stayed nonmigrant correctly predicted, 98 incorrect

gen correct_new_nonmigrant = (correct_nonmigrant==1 & l1_work_us==1)
tab correct_new_nonmigrant
* 0/29 of those who stopped migrating correctly predicted
