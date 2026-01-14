import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utility import find_mean, find_std


if __name__ == "__main__":

    df = pd.read_csv("Data CSV/dataset_train.csv")
    df = df.set_index("Index")

    # Get the 4 unique houses
    houses = df["Hogwarts House"].unique()


    # Take only after the 6 column {hard coded for our data}
    # course_cols: pd.DataFrame = df.columns[6:]
    course_cols = df.select_dtypes(include="number")

    scaler = StandardScaler()

    scaler_df = pd.DataFrame(
        scaler.fit_transform(course_cols), 
        columns=course_cols.columns,
        index=course_cols.index
    )

    house_scores = {}  # empty dictionary

    for h in houses:  # loop over Gryffindor, Slytherin, ...

        # filter rows for this house
        house_data = scaler_df[df["Hogwarts House"] == h]

        # Store it like this
        house_scores[h] = {}
        
        for course in course_cols:
            # clean missing values and convert to NumPy array
            clean_value = house_data[course].dropna().values
            house_scores[h][course] = clean_value

    courses_variability = {}

    for course in course_cols:
        house_mean = []

        for h in houses:
            values = house_scores[h][course]
            if len(values) > 0:
                house_mean.append(find_mean(values))
        
        if len(house_mean) == len(houses):
            variability = find_std(house_mean)
            # variability2 = np.std(house_mean)
            courses_variability[course] = variability

    print(courses_variability)

    # Find most homogeneous course
    best_course = min(courses_variability, key=courses_variability.get)
    print(f"Most homogeneous course = {best_course} "
        f"(variability = {courses_variability[best_course]:.4f})")

    # Plot histograms for all courses
    def plot_histograms(course):
        plt.figure(figsize=(8, 6))
        for h in houses:
            values = house_scores[h][course]
            plt.hist(values, bins=50, alpha=0.5, label=h)

        plt.title(f"Histogram of {course} by House MOST HOMOGENEOUS")
        plt.xlabel("Score")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plot_histograms(best_course)
