import pandas as pd

filename = 'Strawberry Greenhouse Environmental Control Dataset(version2).csv'
df = pd.read_csv(filename, encoding='latin1', sep=';', decimal=',', parse_dates=['Timestamp'], dayfirst=True, index_col='Timestamp')
df.columns = [c.replace('"', '').strip() for c in df.columns]
print("All Columns:", df.columns.tolist())

outdoor_cols = [c for c in df.columns if 'Outdoor' in c or 'Solar' in c]
print("Outdoor Cols (Detected):", outdoor_cols)
