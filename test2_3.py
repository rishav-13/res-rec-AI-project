import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import requests
from json.decoder import JSONDecodeError

# Load data and models
df = pickle.load(open('restaurants.pkl', 'rb'))
recommendations = pickle.load(open('similarity.pkl', 'rb'))

unique_cuisines_df = pickle.load(open('Cuisine.pkl', 'rb'))
unique_cuisines_list = unique_cuisines_df['Cuisine'].tolist()

def recommend_restaurants(df, preferred_cuisines, budget, min_rating, choice, choice2):
        
    veg = choice
    if(veg == "Yes"):
        df = df[df['isVegOnly'] == 1]
    elif(veg == "No"):
        df = df[df['isVegOnly'] == 0]
    else:
        df = df
    if choice2 == "Seating":
        df_filtered = df[(df['AverageCost'] <= budget) & (df['isIndoorSeating'] == 1) & (df['Dinner Ratings'] >= min_rating)]
    elif choice2 == "Order":
        df_filtered = df[(df['AverageCost'] <= budget) & (df['IsHomeDelivery'] == 1) & (df['Delivery Ratings'] >= min_rating)]
    else:
        df_filtered = df[(df['AverageCost'] <= budget) & (df['Dinner Ratings'] >= min_rating)]
    print(df_filtered)
    if df_filtered.empty:
        return "None"

    df_filtered['Cuisine'] = df_filtered['Cuisines'].apply(lambda x: ' '.join(cuisine.lower() for cuisine in str(x).split(', ')))

    df_filtered['Cuisine'] = df_filtered['Cuisine'].str.lower()

    df_filtered = df_filtered[df_filtered['Cuisine'].apply(lambda x: any(cuisine.lower() in x for cuisine in preferred_cuisines))]

    count = CountVectorizer(stop_words='english')
    count_matrix = count.fit_transform(df_filtered['Cuisine'])

    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    df_filtered = df_filtered.reset_index()
    indices = pd.Series(df_filtered.index, index=df_filtered['Name'])

    df_filtered['Unique_ID'] = range(1, len(df_filtered) + 1)
    df_filtered['Unique_ID'] = df_filtered['Unique_ID'].astype(int)

    def get_recommendations(unique_id, cosine_sim=cosine_sim):

        idx = int(unique_id - 1)
        sim_scores = cosine_sim[idx]
        threshold = np.mean(sim_scores)
        scores = sim_scores

        if any(score > threshold for score in sim_scores):
            sim_scores_with_indices = list(enumerate(sim_scores))
            sim_scores_with_indices = sorted(sim_scores_with_indices, key=lambda x: x[1], reverse=True)
            sim_scores_with_indices = sim_scores_with_indices[1:11]
            restaurant_indices = [i[0] for i in sim_scores_with_indices]

            location = df_filtered[['Full_Address']].iloc[restaurant_indices]
            print(location)
            return df_filtered[['Name', 'URL', 'Full_Address', 'AverageCost', 'Cuisines']].iloc[restaurant_indices]
        else:
            return None

    recommendations = get_recommendations(df_filtered[df_filtered['Cuisine'].apply(lambda x: any(cuisine.lower() in x for cuisine in preferred_cuisines))]['Unique_ID'].iloc[0])

    return recommendations
def get_cuisine_image_urls(cuisine, count=10):
    try:
        # Check if the cuisine represents a country or region
        country_cuisines = ['chinese', 'italian', 'mexican','north indian','south indian','tibetan','bengali','bangladeshi','misti','arabian','hydrabadi','american','rajasthani']  # Add more as needed

        if cuisine.lower() in country_cuisines:
            # Use Pexels API for country-specific cuisines
            pexels_api_key = "q4ZkpwzWplApQpRV08Ij1ElVPasBWn1XV6P4TKesjjCPuQTXlPiDh67A"  # Replace with your Pexels API key
            url = "https://api.pexels.com/v1/search"
            params = {
                "query": f"{cuisine} food",  # Include the term "food" in the query
                "per_page": count
            }
            headers = {"Authorization": pexels_api_key}

            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

            data = response.json()

            if response.status_code == 200 and data.get('photos'):

                image_urls = [photo['src']['original'] for photo in data['photos']]
                return image_urls
            else:
                return ["https://source.unsplash.com/50X60/?food"] * count

        else:
            unsplash_access_key = "K4CMJklp2XVJjW6GUXnzb9Wxavg9TkxLtFf3v9g3zSA"  
            url = "https://api.unsplash.com/search/photos"
            params = {
                "query": cuisine,
                "client_id": unsplash_access_key,
                "per_page": count
            }

            response = requests.get(url, params=params)
            response.raise_for_status()  

            data = response.json()

            if response.status_code == 200 and data.get('results'):
                image_urls = [result['urls']['regular'] for result in data['results']]
                return image_urls
            else:
                return ["https://source.unsplash.com/50X60/?food"] * count

    except requests.exceptions.HTTPError as errh:
        # Handle HTTP errors
        st.warning(f"HTTP Error: {errh}")
        return ["https://source.unsplash.com/50X60/?food"] * count
    except Exception as e:
        # Handle other exceptions
        st.warning(f"Error: {e}")
        return ["https://source.unsplash.com/50X60/?food"] * count
# Streamlit UI
st.set_page_config(page_title="Search Your Similar Restaurants")

title_style = "font-size: 40px; color: pale-blue; font-family: Georgia, serif;"
st.markdown(f'<h1 style="{title_style}">RESIDERELISH</h1>', unsafe_allow_html=True)
dishes_list = ["Burger", "Pizza", "Ice Cream", "Juice", "Sandwich", "Mexican", "Italian", "Chinese", "Fast Food","Momos","Kebab","Wraps","Salad","Healty Food","Cafe","Hot Dogs","Roast Chicken","Bakery","Misti","Arabian","Bengali"]
#Streamlit UI
# Randomly select 3 dishes
random_dishes = np.random.choice(dishes_list, size=3, replace=False)
dish_images = []

# Display images and names
for dish in random_dishes:
    cuisine_image_urls = get_cuisine_image_urls(dish, count=1)
    dish_images.extend(cuisine_image_urls)

# Create a layout with columns
col1, col2, col3 = st.columns(3)

# Display images and names in each column
with col1:
    st.image(dish_images[0], width=200, caption=random_dishes[0])
    

with col2:
    st.image(dish_images[1], width=200, caption=random_dishes[1])
    

with col3:
    st.image(dish_images[2], width=200, caption=random_dishes[2])

with st.sidebar:    
    selected_cuisine = st.selectbox("Select Cuisine", df['Cuisines'].str.split(', ').explode().unique())


    options = [1.0, 2.0, 3.0, 4.0]
    rating = st.selectbox("Enter the desired rating:", options)

    budget_input = st.number_input("Enter the budget", min_value=100, max_value=500, value=250, step=1)

    options2 = ["Yes", "No", "Both"]
    choice = st.selectbox("Do you want Veg:", options2)

    options3 = ["Seating", "Order"]
    choice2 = st.selectbox("Select an option:", options3)


if st.button('Show Restaurants'):
    recommendations = recommend_restaurants(df, [selected_cuisine], budget_input, rating, choice, choice2)

    if recommendations is not None:
        st.subheader(f"Recommended Restaurants for {selected_cuisine}")
        # Use Unsplash API to get a collection of food-related images based on cuisine
        cuisine_image_urls = get_cuisine_image_urls(selected_cuisine, count=5)

        for index, row in recommendations.iterrows():
            # Ensure the number of images matches the number of recommended restaurants
            if len(cuisine_image_urls) == 0:
                st.warning("Insufficient images for all recommended restaurants.")
                break

            # Get the corresponding image for the current restaurant (use modulo to wrap around)
            image_url = cuisine_image_urls[index % len(cuisine_image_urls)]

            # Create an expander for each restaurant
            with st.expander(f"{row['Name']}"):
                # Display the selected food-related image
                st.image(image_url, caption=row['Name'], width=200)

                # Display additional restaurant information
                st.markdown(
                    f"**Name:**{row['Name']}<br>"
                    f"**Location:** {row['Full_Address']}<br>"
                    f"**URL:** {row['URL']}<br>"
                    f"**Average Cost:** {row['AverageCost']}<br>"
                    f"**Cuisines:** {row['Cuisines']}",
                    unsafe_allow_html=True
                )

                # Add a "Read More" button
                button_key = f"read_more_{index}"
                if st.button("Read More", key=button_key):
                    # When the button is clicked, you can add more details or actions
                    st.write("Additional details can go here.")

                st.write("---")
    else:
        st.write("No recommendations found.")


html_content = """
<style>
    *,
    *:after {
      box-sizing: border-box;
    }

    h1 {
      font-size: clamp(20px, 15vmin, 20px);
      font-family: sans-serif;
      color: bloack;
      position: relative;
    }

    h1:after {
      content: "";
      position: absolute;
      width: 100%;
      height: 5px;
      background: hsl(130 80% 50%);
      left: 0;
      bottom: 0;
    }
</style>
"""

st.write(html_content, unsafe_allow_html=True)
