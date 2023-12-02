import matplotlib.pyplot as plt
import numpy as np

# Creating a custom probability distribution where numbers between 2-5 have higher probability
# Using a simple piecewise probability distribution for demonstration
def custom_probability_distribution(min, max, probability, prefferedMin, prefferedMax):
    random_number = np.random.rand()
    print("random number: {}".format(random_number))
    if random_number < probability:  # 70% probability
        return np.random.randint(prefferedMin, prefferedMax)  # higher probability for numbers 2-5
    else:
        # Remaining 50% probability distributed among all numbers from 0-10, excluding 2-5
        choices = list(range(min, prefferedMin)) + list(range(prefferedMax, max))
        return np.random.choice(choices)


def gausianlike_probability_distribution(min_val, max_val, peak, std_dev):
    while True:
        # Generate a number from a Gaussian distribution
        number = np.random.normal(loc=peak, scale=std_dev)

        # Clip the number to the specified range and return if in range
        if min_val <= number <= max_val:
            return int(number)

def plot_number_occurrences(arr):
    # Group numbers by tens
    # Group numbers by tens using list comprehension
    groups = [num // 10 for num in arr]

    # Count occurrences of each group
    counts = [groups.count(i) for i in range(10)]

    # Plotting
    plt.bar(range(10), counts, tick_label=[f"{i*10}-{i*10+9}" for i in range(10)])
    plt.xlabel('Number Groups')
    plt.ylabel('Occurrences')
    plt.title('Occurrences of Number Groups')
    plt.show()
