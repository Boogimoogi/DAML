"""
Q1-a:
In summary, the code first transfer data from "question.csv" , which seems to be a long fromat to a wide format based on "Group" and "Category". 
The code first walk through and extract the data from origin data and create multiple columns for further usage.
And then go over the extracted lists again to mark each category as 1 if it is present in the group, otherwise 0.
Finally, check the absence of category D in these groups. The 'result' variable in the code will counts the number of groups where category D is absent.


"""

## Q1-b
import pandas as pd
df = pd.read_csv("question.csv")

# Use pivot table to create the required DataFrame directly without looping through rows
pivot_df = df.pivot_table(index='Group', columns='Category', aggfunc='size', fill_value=0)

# Convert presence/absence to 1/0 by checking if values are greater than 0 (presence)
categories = ['A', 'B', 'C', 'D', 'E']
for category in categories:
    pivot_df[category] = (pivot_df[category] > 0).astype(int)

# Add "_Count" columns directly
for category in categories:
    pivot_df[f"{category}_Count"] = pivot_df[category]

# Count the groups where category D is absent
result = (pivot_df["D"] == 0).sum()
print(result)
