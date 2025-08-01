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

use "$data/weatherdata.dta", clear
sort village year distancekm
by village year: keep if _n==1 // keep the closest weather station
distinct village year, joint
distinct wsid year, joint
keep wsid village year RENE-Ttemp_std
tempfile yw
save `yw'

cd "$data/daily"
local files: dir . files "*DLY.dta"
clear
append using `files'
cd "$path"

/*
| elementcode | Variable (daily)                     | Units |
| ----------- | ------------------------------------ | ----- |
| **1**       | Precipitation (total)                | mm    |
| **2**       | Evaporation (pan)                    | mm    |
| **3**       | Wind-run                             | km    |
| **5**       | Precipitation (≥ 0.1 mm indicator\*) | (0/1) |
| **18**      | Sunshine duration                    | hours |
| **30**      | Maximum temperature                  | °C    |
| **31**      | Minimum temperature                  | °C    |
| **32**      | Mean temperature                     | °C    |
| **43**      | Relative humidity                    | %     |
*/

gen ym = monthly(yearmonth, "YM")
format ym %tm
gen year = year(dofm(ym))
gen month = month(dofm(ym))
order year month
drop if year<1980 | year>2010

drop ym yearmonth datasetid flag*
distinct year month stationid elementcode, joint

forval i = 1/31 {
	rename value`i' v`i'_e
}

qui reshape wide v*, i(year month stationid) j(elementcode)

foreach var of varlist v* {
	rename `var' `var'_m
}

qui reshape wide v*, i(year stationid) j(month)
rename stationid wsid
distinct year wsid, joint

distinct wsid year
merge 1:m wsid year using `yw'
keep if _merge==3
drop _merge

distinct wsid year village
save "$data/daily_weather_all.dta", replace
