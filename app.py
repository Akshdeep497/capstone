# Importing necessary libraries
#!pip install supabase
from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import re
import sqlite3
from datetime import datetime
from supabase import create_client, Client

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY5"))

# Supabase configuration
SUPABASE_URL = 'https://bxeanvxilpcswieocvej.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ4ZWFudnhpbHBjc3dpZW9jdmVqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzU5MjYxODcsImV4cCI6MjA1MTUwMjE4N30.exoXaJLn1qiuMERvacjEtm_oMEOIWc0mJQR7z59gM3o'
supabase= create_client(SUPABASE_URL, SUPABASE_KEY)

#Function for feature extraction and db storage
def store_indb_feature(prompt, image):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Generate the response from the model
    response = model.generate_content([prompt, image])
    response = response.text.strip()
    response = response[1:-1]
    print(response)

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Insert data into the Supabase table
    data = {
        "timestamp": current_time,
        "brand": response,
        "count": 1
    }
    result = supabase.table("productquantity").insert(data).execute()
    print("Data stored successfully in Supabase!", result)
    return response


def store_indb_freshness(prompt, image):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Generate the response from the model
        response = model.generate_content([prompt, image])
        response = response.text.strip()
        response = response[1:-1]

        # Parse the response string
        parts = response.split(",")
        if len(parts) < 4:
            raise ValueError("Unexpected response format. Expected at least 4 parts.")

        name = parts[0].strip()
        freshness = parts[1].strip()
        days_left = parts[2].strip()
        spoiled = parts[3].strip()

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Insert data into the Supabase table
        data = {
            "timestamp": current_time,
            "name": name,
            "freshness": freshness,
            "days_left": days_left,
            "spoiled": spoiled
        }
        result = supabase.table("freshnessdata").insert(data).execute()
        print("Data stored successfully in Supabase!", result)

        res = f"""- Produce: {name}
- Freshness: {freshness}
- Days left: {days_left}
- Spoiled: {spoiled}"""
        return res
    except Exception as e:
        print(f"An error occurred: {e}")



def store_indb_expiry(prompt, image):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')

    # Generate the response from the model
    response = model.generate_content([prompt, image])
    response = response.text.strip()[1:-1]
    #print(response)

    # Split the response and extract details
    parts = response.split(",")
    name = parts[0].strip()
    useby = parts[1].strip()
    expired1 = parts[2].strip()

    # Calculate days left
    expiry_date = datetime.strptime(useby, '%d/%m/%Y')
    current_date = datetime.now()
    daysleft = (expiry_date - current_date).days

    current_time = current_date.strftime('%Y-%m-%d %H:%M:%S')

    # Prepare data for Supabase
    data = {
        "timestamp": current_time,
        "brand": name,
        "count": 1,
        "expirydate": useby,
        "expired": expired1,
        "expected_lifespan_days": daysleft
    }

    # Insert data into the Supabase table
    result = supabase.table("productquantity").insert(data).execute()
    print("Data stored successfully in Supabase!", result)

    # Return the formatted response
    res = f"""- Brand/Product: {name}
- Expiry Date: {useby}
- Expired: {expired1}
- Days Left: {daysleft} days
"""
    return res


def store_product_quantity(prompt, image):
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest')

        # Generate the response from the model
        response = model.generate_content([prompt, image])
        response = response.text.strip()
        response = response[1:-1]

        product_entries = re.findall(r'([a-zA-Z0-9\s\+\-\&\(\)]+)\s*,\s*(\d+)', response)

        if not product_entries:
            raise ValueError("No valid product entries found in the response.")

        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Prepare data and formatted response
        res_lines = []
        total_count = 0
        for name, quantity in product_entries:
            name = name.strip()[1:]  # Adjust indexing
            quantity = int(quantity.strip())
            total_count += quantity

            # Insert each product into Supabase
            data = {
                "timestamp": current_time,
                "brand": name,
                "count": quantity
            }
            supabase.table("productquantity").insert(data).execute()

            res_lines.append(f"- {name} : {quantity}N")
        res_lines.append(f"- Total : {total_count}N")

        print("Product quantities stored successfully in Supabase!")
        res = "\n".join(res_lines)
        return res
    except Exception as e:
        print(f"An error occurred: {e}")





# Initialize Streamlit app
st.set_page_config(page_title="Flipkart Smart Vision System")
st.header("Flipkart Smart Vision System")

# File uploader for the image
uploaded_file = st.file_uploader("Take/Upload Image", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Display the image with custom width (e.g., 400 pixels)
    st.image(image, caption="Image", width=75)  # Adjust width as needed


#previous prompts before DB implementation
prompts1= {
    "Feature Extraction": "give details such as brand name,product name, and other key features from the packaging material visible in the image. and give output as brand - maggi and so on for all features give every output in bullet points",
    "Expiry Date": "give expiry date/use by/best before as expiry date-(if not mentioned calculate by Manufacture date and best before months), give Manufacture date as Mfg date-, give expired -Yes/No ,give months left as months left-(calculate from expiry or best before date ) give all outputs in bullet points",
    "Counting and Brand Recognition": "give brand and product name and quantity of that product give it in an list for eg 1) maggi noodles - 2N and so on , and in the last give the total number of products in format total quantity - 5N and if there is fruit/vegetable just replace brand name by fruit name give output in bullet points",
    "Freshness Level": "give name and the freshness level of the fruit/vegetable in the image give a name to freshness level eg. banana - ripe , give percentage level of freshness eg Freshness Percentage - 40 percentsign , give edible/not edible give all output in bullet points"
}  

# prompts after DB implementation
prompts = {
    "Freshness_db":"you will be given image of Fruit/vegetable you have ot give output as (name,freshness out of 10,days left before spoiled,spoiled-yes/no) you output should look like this(apple,8,10,no)",
    "ircount_db":"you will be give an image containing different products/fruits/vegetables give output as (brand_and_product_name or item_name,count of the product/item) your output should look like ((oreo,1),(maggi,1),(lays classic,1)) dont give any extra information just give product name and qunatity like i specified",
    "expiry_db":"i will give you an image of product you have to give me its expirydate it can be either useby/expirydate/bestbefore give output as(brand_and_product_name,expirydate(can be in form of useby/expirydate/bestbefore),expired yes/no) your output should look like (maggi,dd/mm/yy,No) if expiry date not visible or you cant determine expiry date your output should be(maggi,NA,NA,NA) and if you cant determine product name but determine expiry date your output should be like (NA,dd/mm/yy,No,20) or if cannot see both give output (NA,NA,NA,NA) you give days left wrong always calculate it properly remember if expiry date is in mm/yyyy convert it to dd/mm/yyyy by replacing dd to 01",
    "feature_db":"i will give you an image of the product give me the brand name and product name your output should look like (oreo biscuits)"
}
  

# Initialize a variable to store the response
response = None

# Create columns to place buttons in a single line
col1, col2, col3, col4 = st.columns(4)

# Handle button clicks within each column
with col1:
    if st.button("Extract Features"):
        
        prompt2=prompts["feature_db"]
        response=store_indb_feature(prompt2,image)


with col2:
    if st.button("Expiry Date"):
        
        prompt2= prompts["expiry_db"]
        
        response=store_indb_expiry(prompt2,image)

with col3:
    if st.button("IR Counting"):
        
        prompt2 = prompts["ircount_db"]
        
        response=store_product_quantity(prompt2,image)


with col4:
    if st.button("Freshness Level"):
        
        prompt2=prompts["Freshness_db"]
        
        response=store_indb_freshness(prompt2,image)

# Display the response below all the buttons
if response:
    st.subheader("Response")
    st.write(response)

