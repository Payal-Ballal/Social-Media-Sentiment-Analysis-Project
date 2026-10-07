# Social Media Sentiment Analysis Dashboard

## Project Overview
An interactive dashboard built with Python and Streamlit to analyze sentiment in social media text. The project uses data processing and natural language processing techniques to classify posts and visualize sentiment patterns.

## Features
- Sentiment classification into Positive, Negative, and Neutral categories
- Interactive sentiment filters and keyword search
- KPI cards and sentiment distribution charts
- VADER compound sentiment score analysis
- Most common word analysis
- CSV export of analyzed posts

## Technologies Used
- Python
- Pandas
- Streamlit
- Matplotlib
- VADER Sentiment Analysis
- CSV data processing

## How It Works
1. Loads social media text from a CSV dataset.
2. Cleans the text for analysis.
3. Uses existing sentiment labels where available.
4. Applies VADER sentiment analysis to posts with missing labels.
5. Displays interactive visualizations and allows filtered results to be exported.

## Installation and Setup

1. Clone or download this repository.
2. Install the dependencies:

   `pip install -r requirements.txt`

3. Run the application:

   `python -m streamlit run app.py`

4. Open the local URL displayed in the terminal.

## Data and Limitations
The dashboard currently analyzes a downloaded dataset and does not collect live posts from X/Twitter. Original dataset labels and VADER predictions are different classification sources. The dataset does not provide actual post timestamps, so time-based trend analysis is not available.

## Purpose
This project demonstrates practical skills in Python data processing, sentiment analysis, interactive dashboard development, and data visualization.