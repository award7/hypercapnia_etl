# Hypercapnia ETL
ETL pipeline for hypercapnia data processing. 

## Motivation
To develop an automated and transparent method to analyze hypercapnia-mediated
changes in middle cerebral artery blood flow velocity as described 
[here](https://clinicaltrials.gov/ct2/show/NCT04154865?term=adolescents%2C+exercise&recrs=am&cond=Insulin+Resistance&draw=2&rank=2). 

## Preprocessing
Data files are exported from data acquisition software (e.g. LabChart) and 
loaded to common staging area in filesystem (e.g. file server).

## Assumptions
To use as-is, there are multiple assumptions:
- Column names
- `.txt` file extensions on preprocessed files
- Timing labels
- `#NUM!` flags
- 