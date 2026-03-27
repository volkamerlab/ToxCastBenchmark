with open('morgan_none.txt', 'r') as morgan_file:
    morgan_assays = set(morgan_file.read().splitlines())
with open('considered_hormone_assay_names.txt', 'r') as all_assay_file:
    all_assays = set(all_assay_file.read().splitlines())

print(all_assays - morgan_assays)