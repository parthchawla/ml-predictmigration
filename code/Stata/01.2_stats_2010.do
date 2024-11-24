* ------------------------------------------------------------------------------
/*
*        Created by: Parth Chawla
*        Created on: Nov 24, 2024
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
use "$data/MexMigData.dta", clear

* Make sure we have our experience and transition variables
gen us_experience = ""
replace us_experience = "No experience" if totalus == 0
replace us_experience = "1-3 years" if totalus >= 1 & totalus <= 3
replace us_experience = "4-7 years" if totalus >= 4 & totalus <= 7
replace us_experience = "8-15 years" if totalus >= 8 & totalus <= 15
replace us_experience = "16+ years" if totalus >= 16 & totalus != .
tab us_experience

* Create lagged migration status
order ind year work_us
sort ind year
xtset ind year
forval i = 1/5 {
	local y = 2010 - `i'
	display "Year: `y'"
	gen work_us_`y' = L`i'.work_us
}

keep if year == 2010
rename work_us work_us_2010
keep ind work_us* totalus everus us_experience everus

egen past5tot = rowtotal(work_us_2009 work_us_2008 work_us_2007 work_us_2006 work_us_2005)
gen past5once_us = (past5tot > 0)

tab past5tot
tab past5once_us

* Create a sequence indicator
// gen pattern = ""
// by ind: replace pattern = string(work_us_2010)+string(work_us_2009)+string(work_us_2008) ///
// +string(work_us_2007)+string(work_us_2006)+string(work_us_2005)
// tab pattern



