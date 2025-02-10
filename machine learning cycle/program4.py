import csv
import pandas as pd

data = [
    ["SN","Name","Country","Contribution","Year"],
    [1,"Linus Torvalds","Finland","Linux-Kernel",1991],
    [2,"Tim Berners-Lee","TimBerners-Lee","World Wide Web",1990],
    [3,"Guido van Rossum" ,"Netherlands","Python",1991]

       ]

filename = "dataset/scientist.csv"
with open(filename,mode='w',newline = '') as file:
 writer = csv.writer(file)

 writer.writerows(data)

 print("Successfully written")
print("............................................")
df = pd.read_csv("dataset/scientist.csv")

print(df.head())

