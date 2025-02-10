'''Given a file “auto.csv” of automobile data with the fields origin, company name,
year, num-of-cylinders, displacement, mileage, horsepower, weight and acceleration, write Python codes using Pandas to
----Clean and Update the CSV file
----Print total cars of all companies
----Find the average mileage of all companies'''
import pandas as pd

df = pd.read_csv('dataset\Automobile.CSV')


print(df.describe())
print("....................................................................................")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

df.reset_index(drop=True,inplace=True)
print(df.describe())
print("......................................................................................")
print(df.head())
print("......................................................................................")
df[["company","name"]] = df["name"].str.split(n=1,expand = True)
company_replacements = {
    'mercedes-benz': 'mercedes',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'maxda': 'mazda',
    'chevroelt': 'chevrolet',
    'chevy': 'chevrolet'
}

df['company'] = df['company'].replace(company_replacements)

df.to_csv("dataset/auto.CSV",index = False)
print(".......................................................................................")

ef = pd.read_csv("dataset/auto.CSV")
print("....................................auto.csv.....................................")
print(ef.head())
print("..........................................................................................")
print(ef.describe())
ef.dropna(inplace=False)
ef.drop_duplicates(inplace=False)
ef.reset_index(drop=True,inplace=True)
print("...........................................................................................")
print("...........................................................................................")
car_count = ef.groupby("company").size()
total_car_count = car_count.sum()
print("......................... number of cars by company.....................................")
print(car_count)
print(".........................................................................................")
print(f"Total number of cars by all company : {total_car_count}")
print(".........................................................................................")
sum_of_mileage = ef['mpg'].sum()
number_of_entry = len(ef['mpg'])
print("Average mileage of all companies =",sum_of_mileage/number_of_entry)
