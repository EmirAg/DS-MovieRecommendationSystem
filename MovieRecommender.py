#Imports
import time
import pandas as pd

#Reading Files
df1=pd.read_csv('tmdb_5000_credits.csv')
df2=pd.read_csv('tmdb_5000_movies.csv')

#Defining Columns
df1.columns = ['id','tittle','cast','crew']
df2= df2.merge(df1,on='id')

#Popularity-Based-Method------------------------------------------------------------------------------------------------
#C is the mean vote across the whole report
C= df2['vote_average'].mean()
#m is the minimum votes required to be listed in the chart
m= df2['vote_count'].quantile(0.9)
#Now filter out the movies that qualify for the chart
q_movies = df2.copy().loc[df2['vote_count'] >= m]

#weighted rating function
def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return round((v/(v+m) * R) + (m/(m+v) * C), 1)

# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Getting valid number of movies from user
def get_num_of_movies():
    while True:
        user_input = input("Enter the number of movies you wish to display: ")
        # Convert the user input to an integer
        try:
            num_of_movies = int(user_input)
            # Check if the number is greater than or equal to 1
            if num_of_movies >= 1:
                # If the condition is met, break out of the loop and return the input
                break
            else:
                print("Please enter a number greater than or equal to 1")
                print55()
        except ValueError:
            print("Please enter a valid integer")
            print55()
    return num_of_movies

#Displaying the desired number of movies
def display_top_movies(num):
    """
    Display the top movies based on user-defined number.

    Parameters:
    - num (int): Number of top movies to display.
    """
    # Assuming q_movies is a DataFrame you already have
    result = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(num)
    # Print the result
    print(f"Here are the top {num} movies:")
    print55()
    print(result)
    print55()
#Popularity-Based-Method-END--------------------------------------------------------------------------------------------

#Content-Based-Method---------------------------------------------------------------------------------------------------
#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df2['overview'] = df2['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df2['overview'])

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df2.index, index=df2['title']).drop_duplicates()

def get_recommendations_by_title(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movie titles with each title on a separate line
    return '\n'.join(df2['title'].iloc[movie_indices])

def launch_content_based():
    title = get_movie_title()
    print(f"~Here are movies similar to ({title})")
    print(get_recommendations_by_title(title))
    print55()

def launch_content_based_init():
    try:
        launch_content_based()
    except KeyError:
        print("Movie Not Found.\nPlease try again after checking movie title...")
        print55()
        launch_content_based_init()
#Content-Based-Method-END-----------------------------------------------------------------------------------------------

#Collaborative-Filtering------------------------------------------------------------------------------------------------

#Collaborative-Filtering-END--------------------------------------------------------------------------------------------

#General-Methods--------------------------------------------------------------------------------------------------------
def launch_program():
    print("+" * 55)
    print("       Welcome to Movie Recommendation System!")
    print55()

def print55():
    print("-" * 55)

#Choosing A method for recommending:
def choose_recc_method():
    while True:
        user_in = input("Choose Recommendation Method By Inputting Its Number: \n1: Most Popular"
                        "\n2: Similar Movies By Movie Title\n3: By Collaborative Filtering"
                        "\n0: To Exit...\n" + "-" * 55 + "\nYour Choice: ")
        print55()
        try:
            user_in_int = int(user_in)
            if user_in_int in [0, 1, 2, 3]:
                return user_in_int
            else:
                print("Please choose a valid number!")
                print55()
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            print55()

def get_movie_title():
    movie_title = input('Enter movie title: ')
    print55()
    if movie_title.isupper():
        return movie_title
    else:
        return movie_title.title()
#General-Methods-END----------------------------------------------------------------------------------------------------

#Main-------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    launch_program()
    # Choosing a method for recommendation
    while True:
        user_choice = choose_recc_method()
        if user_choice == 1:
            print("Recommendation By Popularity...")
            num_movies = get_num_of_movies()
            display_top_movies(num_movies)
        elif user_choice == 2:
            print("Recommendation By Content Based...")
            launch_content_based_init()
        elif user_choice == 3:
            print("Recommendation By Collaborative Filtering...")
            print("Coming Soon...")
            print55()
        elif user_choice == 0:
            print("Good Bye!")
            print("+" * 55)
            time.sleep(1)
            exit()
#End-Main---------------------------------------------------------------------------------------------------------------