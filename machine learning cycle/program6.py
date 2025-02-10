import numpy as np
import pandas as pd

# Calling the dataset
filename = "dataset/data1.csv"
df = pd.read_csv(filename)
print(df.head())

# Create a new DataFrame with the relevant columns
data = pd.DataFrame({
    "House": df["House"],
    "Furniture": df["Furniture"],
    "Rooms": df["No.rooms"],
    "kitchen": df["New kitchen"],
    "Acceptable": df["Acceptable"]
})

# Function to calculate entropy
def calculate_entropy(labels):
    unique, count = np.unique(labels, return_counts=True)
    probabilities = count / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Function to calculate information gain
def information_gain(data, split_attribute, t="Acceptable"):
    total_entropy = calculate_entropy(data[t])  # Calculate total entropy of the target column
    values, count = np.unique(data[split_attribute], return_counts=True)  # Get unique values and counts in the split attribute
    weight = 0
    
    # Loop over each value of the split attribute
    for i, v in enumerate(values):
        prob = count[i] / sum(count)  # Probability of each value
        subset = data[data[split_attribute] == v]  # Subset of data where split_attribute == v
        weight_entropy = prob * calculate_entropy(subset[t])  # Weighted entropy for each value of the split attribute
        weight += weight_entropy  # Add the weighted entropy to the total
    
    return total_entropy - weight  # Information gain is the difference between the total entropy and the weighted entropy


def id3(data, features, target="Acceptable"):
    # If all instances have the same label, return that label
    if len(np.unique(data[target])) == 1:
        return np.unique(data[target])[0]
    
    # If there are no features left to split on, return the majority label
    if len(features) == 0:
        return np.unique(data[target], return_counts=True)[0][0]
    
    # Calculate information gain for each feature and choose the best feature to split on
    ig_values = {feature: information_gain(data, feature, target) for feature in features}
    best_feature = max(ig_values, key=ig_values.get)
    
    # Create a tree node
    tree = {best_feature: {}}
    
    # Remove the best feature from the list of features to prevent re-splitting on it
    remaining_features = [f for f in features if f != best_feature]
    
    # Split the dataset based on the best feature and recursively build the tree
    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset, remaining_features, target)
    
    return tree
# Target variable
target = df["Acceptable"]

# Calculate initial entropy of the target variable
initial_entropy = calculate_entropy(target)
print(f"Initial entropy = {initial_entropy}")

# Features to evaluate for information gain
features = ["Furniture", "Rooms", "kitchen"]

# Calculate information gain for each feature
for f in features:
    ig = information_gain(data, f)
    print(f"Information gain for {f}: {ig}")

tree = id3(data, features)
print("Decision Tree:")
print(tree)


