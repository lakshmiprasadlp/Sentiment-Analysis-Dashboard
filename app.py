import pandas as pd
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# App title
st.title("Sentiment Analysis Dashboard")
st.write("Upload a CSV file containing customer reviews and analyze sentiments.")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())
    
    # Check for 'Review Text' column
    if 'Review Text' not in data.columns:
        st.error("The uploaded file must contain a 'Review Text' column.")
    else:
        # Sentiment analysis
        st.write("Performing Sentiment Analysis...")
        data['Sentiment'] = data['Review Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        data['Sentiment Label'] = data['Sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        st.write(data[['Review Text', 'Sentiment Label']].head())
        
        # Sentiment distribution
        sentiment_counts = data['Sentiment Label'].value_counts()
        st.write("Sentiment Distribution:")
        st.bar_chart(sentiment_counts)
        
        # Word cloud for positive and negative reviews
        st.write("Word Clouds:")
        positive_text = " ".join(data[data['Sentiment Label'] == 'Positive']['Review Text'])
        negative_text = " ".join(data[data['Sentiment Label'] == 'Negative']['Review Text'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Positive Words")
            positive_wc = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
            plt.imshow(positive_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
        
        with col2:
            st.subheader("Negative Words")
            negative_wc = WordCloud(width=400, height=300, background_color='white').generate(negative_text)
            plt.imshow(negative_wc, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
