import csv
import pandas as pd
import matplotlib.pyplot as plt
data = [
    ["month_number", "facecream", "facewash", "toothpaste","bathingsoap","shampoo","moisturizer","total_units","total_profit"],
    [1, 2500, 1500, 5200, 9200, 1200, 1500, 20900, 211000],
    [2, 2630, 1600, 5100, 6100, 1400, 1550, 18380, 183400],
    [3, 2140, 1340, 4550, 9550, 1330, 1290, 20200, 217600],
    [4, 3400, 1130, 5870, 8870, 1640, 1700, 22610, 241300],
    [5, 3600, 1740, 4560, 7760, 1560, 1200, 20420, 217000],
    [6, 2760, 1555, 4890, 7490, 1890, 1350, 19935, 212500],
    [7, 2980, 1120, 4500, 9300, 2100, 1840, 21840, 224700],
    [8, 3700, 1400, 4780, 9900, 2340, 2150, 24270, 260800],
    [9, 3540, 1780, 4560, 8870, 2260, 2000, 23010, 248500],
    [10, 2900, 1550, 5200, 9560, 1780, 1890, 22880, 236400],
    [11, 3200, 1430, 5120, 8540, 1930, 2100, 22320, 229500],
    [12, 3500, 1800, 4700, 7600, 2340, 1950, 21890, 225400]
]

filename = "dataset/company.csv"

with open(filename,mode='w',newline ='') as file:
    writer = csv.writer(file)

    writer.writerows(data)

    print("Successfully written")
print("...........................................................................................")

df = pd.read_csv("dataset/company.csv")
month = df['month_number']
sales_toothpaste = df['toothpaste']
plt.scatter(month,sales_toothpaste,color = "blue",marker='o', label = 'toothpaste')
plt.xlabel('month')
plt.ylabel('unit solds')
plt.title("Monthly Sales")

plt.legend()

plt.show()

print("............................................................................................")

sales_facecream = df['facecream']
sales_facewash = df['facewash']

plt.bar(month,sales_facecream,color = "blue",label = "facecream")
plt.bar(month,sales_facewash,color = "Red",label = "facewash" ,alpha = 0.7)


plt.xlabel("Month")
plt.ylabel("unit solds")
plt.title("Monthly sales")

plt.legend()
plt.show()
print(".............................................................................................")

sales_bathingsoap = df['bathingsoap'].sum()
sales_shampoo = df['shampoo'].sum()
sales_moisturizer = df['moisturizer'].sum()
sales_toothpaste_sum = df['toothpaste'].sum()
sales_facecream_sum = df['facecream'].sum()
sales_facewash_sum = df['facewash'].sum()
print(f"{sales_bathingsoap},{sales_shampoo},{sales_moisturizer},{sales_toothpaste_sum },{sales_facecream_sum},{sales_facewash_sum }")

categories = ["facecream", "facewash", "toothpaste","bathingsoap","shampoo","moisturizer"]
total_sales = [sales_facecream_sum,sales_facewash_sum,sales_toothpaste_sum,sales_bathingsoap,sales_shampoo,sales_moisturizer]

plt.pie(total_sales,labels=categories,autopct="%1.1f%%",startangle=140, colors=['red', 'blue', 'green', 'purple', 'orange', 'cyan'])

plt.title("Sales Distribution of Products")

# Display the Pie Chart
plt.show()

