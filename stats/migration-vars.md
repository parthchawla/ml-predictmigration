# Key Migration Variables

## Location-Based Work Indicators (Binary)
- `work_loc`: Worked locally (1/0)
- `work_mx`: Worked elsewhere in Mexico (1/0)
- `work_us`: Worked in US (1/0)
- `samestate`: Migrated within Mexican state (1/0)
- `locmig`: Worked both locally and elsewhere in Mexico (1/0)

## US Migration Details
1. Duration & Frequency:
   - `countus`: Number of consecutive years working in US
   - `yearus`: Years of trip to work in US
   - `totalus`: Total years worked in US 1980-2007
   - `everus`: Ever worked in the US between 1980-2007
   - `usperm`: In US in 2007 (1/0)
   - `timeinus`: Years of a completed trip to US

2. Type of Work in US:
   - `us_ag_sal_`: Agricultural salary work
   - `us_nonag_sal_`: Non-agricultural salary work
   - `us_ag_own_`: Agricultural self-employed
   - `us_nonag_own_`: Non-agricultural self-employed
   - `usstate`: US state worked in
   - `ussector`: US sector of work

## Mexico Internal Migration Details
1. Duration & Frequency:
   - `countmx`: Number of consecutive years working elsewhere in Mexico
   - `yearmx`: Years of trip to work elsewhere in Mexico
   - `totalmx`: Total years worked elsewhere in Mexico 1980-2007
   - `evermx`: Ever worked elsewhere in Mexico between 1980-2007
   - `mxperm`: Elsewhere in Mexico in 2007 (1/0)
   - `timeinmex`: Years of a completed trip within Mexico

2. State-Level Migration:
   - `countmxstate`: Consecutive years working elsewhere in Mexican state of origin
   - `yearmxstate`: Years of trip to work in Mexican state of origin
   - `totalmxstate`: Total years worked elsewhere in Mexican state 1980-2007
   - `evermxstate`: Ever worked elsewhere in Mexican state between 1980-2007
   - `mxpermstate`: In Mexican state of origin in 2007 (1/0)

3. Type of Work in Mexico:
   - `mx_ag_sal_`: Agricultural salary work elsewhere in Mexico
   - `mx_nonag_sal_`: Non-agricultural salary work elsewhere in Mexico
   - `mx_ag_own_`: Agricultural self-employed elsewhere in Mexico
   - `mx_nonag_own_`: Non-agricultural self-employed elsewhere in Mexico
   - `mxstate`: Mexican state worked in
   - `mxsector`: Sector elsewhere in Mexico

## Local Work Details
- `loc_ag_sal_`: Agricultural salary work in community
- `loc_nonag_sal_`: Non-agricultural salary work in community
- `loc_ag_own_`: Agricultural self-employed in community
- `loc_nonag_own_`: Non-agricultural self-employed in community
- `locsector`: Local sector of work
