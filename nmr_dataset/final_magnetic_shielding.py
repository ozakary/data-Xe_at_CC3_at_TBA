import pandas as pd

# Read the CSV files
file1 = pd.read_csv('./BHandHLYP_SVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv')
file2 = pd.read_csv('./PBE_TZVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv')
file3 = pd.read_csv('./PBE_SVP/FINISHED/dataset_schnet_shielding_tensors/magnetic_shielding_tensors_modified.csv')

# Ensure the files have the same number of rows and same first and second columns
assert (file1.shape[0] == file2.shape[0] == file3.shape[0]), "Files must have the same number of rows"
assert (file1.columns[0] == file2.columns[0] == file3.columns[0]), "First column headers must be the same"
assert (file1.columns[1] == file2.columns[1] == file3.columns[1]), "Second column headers must be the same"

# Extract the first two columns
first_two_columns = file1.iloc[:, :2]

# Perform the operation for each column starting from the third column
result_columns = []
for col in file1.columns[2:]:
    result_col = file1[col] + file2[col] - file3[col]
    result_columns.append(result_col)

# Combine the result with the first two columns
result_df = pd.concat([first_two_columns] + result_columns, axis=1)
result_df.columns = file1.columns  # Set the column names to match the input files

# Write the result to a new CSV file
result_df.to_csv('magnetic_shielding_tensors.csv', index=False)

print("Operation completed and result saved to 'magnetic_shielding_tensors.csv'")
