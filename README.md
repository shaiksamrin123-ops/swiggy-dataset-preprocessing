# Swiggy Dataset Exploration & Preprocessing

## Project Overview
This project focuses on exploring, cleaning, and preparing a Swiggy restaurant dataset for future machine learning applications using Python.

## Objectives
- Explore dataset structure and data quality.
- Handle missing and invalid values.
- Convert price, rating, and rating-count fields into numeric formats.
- Identify and remove duplicate records.
- Detect price outliers using the IQR method.
- Visualize restaurant ratings, prices, cuisines, and locations.

## Tools & Technologies
- Python
- Pandas
- Matplotlib
- Jupyter Notebook

## Dataset Summary
- Original records: 140,657
- Cleaned records: 139,321
- Columns: 10
- Exact duplicate rows removed: 1,336

## Preprocessing Performed
1. Inspected data types, missing values, and duplicates.
2. Converted ratings and rating counts into numeric values.
3. Extracted numeric prices from currency-formatted text.
4. Filled missing cuisine and area values with `Unknown`.
5. Marked restaurants with zero offers as `No Offer`.
6. Removed exact duplicate rows.
7. Standardized column names to lowercase snake_case.
8. Validated rating ranges and remaining missing values.
9. Identified price outliers without automatically deleting them.
10. Created data visualizations.

## Key Findings
- 75.99% of records have rating information available.
- 24.01% have both rating and rating-count information missing.
- Restaurant ratings range from 1 to 5.
- The IQR method flagged 11,069 price outliers for further investigation.

**Note:** Rating-count labels such as `1K+ ratings` were converted to approximate lower-bound values, not exact counts. Missing rating values were retained rather than filled with zero.

## Output
The cleaned dataset was exported as `swiggy_cleaned.csv`.

## Future Improvements
- Investigate extreme price values.
- Explore relationships between price, cuisine, location, and ratings.
- Define a suitable machine learning objective before model training.

