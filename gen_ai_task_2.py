"""
@Author: Venkatesh
@Date: 21-11-2024
@Last Modified by: Venkatesh
@Last Modified time: 21-11-2024
@Title: Python program to perform Gen AI tasks using Gemini on identify reviews like positive or negative or neutral
"""

import os
import csv
import time
from dotenv import load_dotenv
import google.generativeai as genai

def read_reviews(file_path, delimiter="---END OF REVIEW---"):
    """
    Description:Reads the reviews from a text file.
    
    Parameters:
        file_path (str): Path of the text file.
        delimiter (str): Delimiter separating reviews in the file.
        
    Returns:
        list: List of non-empty reviews.
    """
    with open(file_path, 'r') as file:
        reviews = file.read().split(delimiter)
    return [review.strip() for review in reviews if review.strip()]

def extract_review_info(review):
    """
    Description: Extracts product name and review text from a review.
    
    Parameters:
        review (str): The full review text.
        
    Returns:
        tuple: A tuple containing the product name and review text.
    """
    product = ""
    review_text = ""

    for line in review.split("\n"):
        if line.startswith("Product:"):
            product = line.split(":", 1)[1].strip()
        elif line.startswith("Review:"):
            review_text = line.split(":", 1)[1].strip()
    
    return product, review_text

def analyze_sentiment(review_text, chat_session):
    """
    Description: Analyzes the sentiment of a review using the Gemini model.
    
    Parameters:
        review_text (str): The review text to analyze.
        chat_session: The chat session with the Gemini model.
        
    Returns:
        tuple: A tuple containing the sentiment (str) and the generated reply (str).
    """
    # Refined instruction for sentiment analysis
    sentiment_response = chat_session.send_message(
        f"Classify the sentiment of this review as Positive, Negative, or Neutral: {review_text}"
    )
    sentiment = sentiment_response.text.strip()

    # Check if the sentiment response matches our expectations (e.g., Positive, Negative, Neutral)
    if sentiment not in ['Positive', 'Negative', 'Neutral']:
        sentiment = "Neutral"  # Default to neutral if the classification is unexpected.

    # Generate a reply based on the sentiment
    reply_response = chat_session.send_message(
        f"Write a reply to this review based on the sentiment: {sentiment}. Review: {review_text}"
    )
    reply = reply_response.text.strip()

    return sentiment, reply

def guess_product(review_text, chat_session):
    """
    Description: Guess the product category based on the review text.
    
    Parameters:
        review_text (str): The review text to analyze.
        chat_session: The chat session with the Gemini model.
        
    Returns:
        str: The guessed product category.
    """
    product_response = chat_session.send_message(
        f"Guess the product name based on this review in one word: {review_text}"
    )
    return product_response.text.strip()

def save_to_csv(data, csv_file):
    """
    Description: Save the processed review data to a CSV file.
    
    Parameters:
        data (list): List of processed data (rows) to save.
        csv_file (str): The path to save the CSV file.
        
    Returns:
        None
    """
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Original Product", "Guessed Product", "Review", "Sentiment", "Reply"])
        writer.writerows(data)

def process_reviews(file_path, csv_file, chat_session, delimiter="---END OF REVIEW---"):
    """
    Descption: Process reviews by reading from a file, analyzing them, and saving results to a CSV.
    
    Parameters:
        file_path (str): The path to the text file containing reviews.
        csv_file (str): The path to save the processed reviews.
        chat_session: The chat session with the Gemini model.
        delimiter (str): The delimiter separating reviews in the file.
        
    Returns:
        None
    """
    reviews = read_reviews(file_path, delimiter)
    data = []

    for review in reviews:
        original_product, review_text = extract_review_info(review)
        guessed_product = guess_product(review_text, chat_session)
        sentiment, reply = analyze_sentiment(review_text, chat_session)

        data.append([original_product, guessed_product, review_text, sentiment, reply])
        time.sleep(2)  # Sleep to avoid rate-limiting

    save_to_csv(data, csv_file)

def main():
    """
    Main function to execute the review processing pipeline:
    
    - Configures the Gemini API.
    - Initializes a chat session.
    - Processes reviews from an input file and saves results to a CSV file.
    """
    try:
        # Load environment variables and configure Gemini
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            raise ValueError("API key not found in environment variables.")

        genai.configure(api_key=api_key)

        # Create model configuration
        generation_config = {
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        # Initialize chat session with the Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash", generation_config=generation_config)
        chat_session = model.start_chat(history=[])

        # Process reviews and save results to CSV
        process_reviews('reviews.txt', 'transformed_reviews.csv', chat_session)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
