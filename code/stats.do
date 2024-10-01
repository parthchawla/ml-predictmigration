
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
global output "output"

use "$data/MexMigData.dta", clear

egen id = group(id1s2 numc)
order id year
sort id year

* Create lag variables to check previous migration status
by id: gen work_us_lag1 = work_us[_n-1] if year == 2010
by id: gen work_us_lag2 = work_us[_n-2] if year == 2010
by id: gen work_us_lag3 = work_us[_n-3] if year == 2010
by id: gen work_us_lag4 = work_us[_n-4] if year == 2010

* Focus on individuals in 2010
keep if year == 2010

* Total number of migrants in 2010
count if work_us == 1
local total_migrants = r(N)

* Number of 2010 migrants who also migrated in 2009
count if work_us == 1 & work_us_lag1 == 1
local migrants_2009 = r(N)

* Number of 2010 migrants who also migrated in 2008
count if work_us == 1 & work_us_lag2 == 1
local migrants_2008 = r(N)

* Number of 2010 migrants who also migrated in 2007
count if work_us == 1 & work_us_lag3 == 1
local migrants_2007 = r(N)

* Number of 2010 migrants who also migrated in 2006
count if work_us == 1 & work_us_lag4 == 1
local migrants_2006 = r(N)

* Calculate percentages
local pct_2009 = (`migrants_2009' / `total_migrants') * 100
local pct_2008 = (`migrants_2008' / `total_migrants') * 100
local pct_2007 = (`migrants_2007' / `total_migrants') * 100
local pct_2006 = (`migrants_2006' / `total_migrants') * 100

* Display the results
display "Percentage of 2010 migrants who also migrated in 2009: " round(`pct_2009', 0.1) "%"
display "Percentage of 2010 migrants who also migrated in 2008: " round(`pct_2008', 0.1) "%"
display "Percentage of 2010 migrants who also migrated in 2007: " round(`pct_2007', 0.1) "%"
display "Percentage of 2010 migrants who also migrated in 2006: " round(`pct_2006', 0.1) "%"

* Transition probabilities from 2009 to 2010
use "$data/MexMigData.dta", clear
keep if year == 2009 | year == 2010
egen id = group(id1s2 numc)
keep id year work_us

reshape wide work_us, i(id) j(year)

* Create transition categories
gen transition = .
replace transition = 1 if work_us2009 == 0 & work_us2010 == 0
replace transition = 2 if work_us2009 == 0 & work_us2010 == 1
replace transition = 3 if work_us2009 == 1 & work_us2010 == 0
replace transition = 4 if work_us2009 == 1 & work_us2010 == 1

* Label transitions
label define trans 1 "Non-Migrant to Non-Migrant" ///
                   2 "Non-Migrant to Migrant" ///
                   3 "Migrant to Non-Migrant" ///
                   4 "Migrant to Migrant"
label values transition trans

* Tabulate transitions
tabulate transition

* Calculate transition probabilities
tabulate work_us2009 work_us2010, row nofreq

