"""
@Author: Venkatesh
@Date: 21-11-2024
@Last Modified by: Venkatesh
@Last Modified time: 21-11-2024
@Title: Python program to perform Gen AI tasks using Gemini translate sentence

"""

import os
from dotenv import load_dotenv
import google.generativeai as genai
import csv

def read_emails(file_path, delimiter="---END OF EMAIL---"):
    """
    Description:
        Reads emails from a text file and splits them based on the delimiter.

    Parameters:
        file_path (str): Path to the text file containing emails.
        delimiter (str): Delimiter separating emails in the file.

    Returns:
        list: List of email content.
    """
    with open(file_path, 'r') as file:
        emails = file.read().split(delimiter)
    return [email.strip() for email in emails if email.strip()]  # Return non-empty emails


def extract_email_info(email):
    """
    Description:
        Extracts sender, receiver, and body from an email.

    Parameters:
        email (str): The email content.

    Returns:
        tuple: Sender, receiver, and body of the email.
    """
    lines = email.split("\n")
    sender, receiver, body = "", "", ""

    for line in lines:
        if line.startswith("From:"):
            sender = line.split(":", 1)[1].strip()
        elif line.startswith("To:"):
            receiver = line.split(":", 1)[1].strip()
        elif not line.startswith(("Subject:", "From:", "To:")):
            body += line.strip() + " "

    return sender, receiver, body.strip()


def summarize_email(body, chat_session):
    """
    Description:
        Summarizes the body of an email using the Gemini AI model.

    Parameters:
        body (str): The email body content.
        chat_session: The chat session with the Gemini model.

    Returns:
        str: Summarized email content.
    """
    response = chat_session.send_message(f"Summarize the following email: {body}")
    return response.text


def translate_email(text, chat_session):
    """
    Description:
        Translates the given text to German using the Gemini AI model.

    Parameters:
        text (str): The text to be translated.
        chat_session: The chat session with the Gemini model.

    Returns:
        str: Translated text in German.
    """
    response = chat_session.send_message(f"Translate the following text to Telugu: {text}")
    return response.text


def save_to_csv(data, csv_file):
    """
    Description:
        Saves the email data to a CSV file.

    Parameters:
        data (list): The processed email data.
        csv_file (str): The file path to save the CSV.

    Returns:
        None        
    """
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Sender", "Receiver", "Summary (EN)", "Summary (DE)"])
        writer.writerows(data)


def process_emails(file_path, csv_file, chat_session, delimiter="---END OF EMAIL---"):
    """
    Description:
        Processes emails by summarizing and translating their content, then saving the results to a CSV file.

    Parameters:
        file_path (str): The path to the text file containing emails.
        csv_file (str): The file path to save the processed data.
        chat_session: The chat session with the Gemini model.
        delimiter (str): The delimiter separating emails in the file.

    Return:
        None        
    """
    emails = read_emails(file_path, delimiter)
    data = []

    for email in emails:
        sender, receiver, body = extract_email_info(email)
        summary_en = summarize_email(body, chat_session)
        summary_de = translate_email(summary_en, chat_session)
        data.append([sender, receiver, summary_en, summary_de])
    
    save_to_csv(data, csv_file)


def main():
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

        # Process the emails and save results to CSV
        process_emails('email.txt', 'transformed_emails.csv', chat_session)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    main()