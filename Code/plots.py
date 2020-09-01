import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

def plot_year_month(data):
    '''
        Plots two bar charts- statements made by year and by month.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Matplotlib figure
    '''
    fig, axs = plt.subplots(1,2, figsize=(15,5))
    #True/false statements by year
    real_year = data[data['label_fnn'] == 0].groupby('year').count()['statement']
    fake_year = data[data['label_fnn'] == 1].groupby('year').count()['statement']
    #True/false statements by month
    real_month = data[data['label_fnn'] == 0].groupby('month').count()['statement']
    fake_month = data[data['label_fnn'] == 1].groupby('month').count()['statement']
    #Bar charts by year
    width = 0.51
    axs[0].bar(real_year.index -width/2 + .05, real_year, width-.1, label='Real')
    axs[0].bar(fake_year.index +width/2 - .05, fake_year, width-.1, label='Fake')
    #Bar charts by month
    axs[1].bar(real_month.index -width/2 + .05, real_month, width-.1, label='Real')
    axs[1].bar(real_month.index +width/2 - .05, fake_month, width-.1, label='Fake')
    #Setting title, axis, and legends
    axs[0].set_title('Real & Fake Statemnts by Year', fontsize=24)
    axs[0].set_ylabel('Number of Statements', fontsize=20)
    axs[0].set_xlabel('Year', fontsize=20)
    axs[0].legend(loc='best')
    axs[1].set_title('Real & Fake Statemnts by Month', fontsize=24)
    axs[1].set_ylabel('Number of Statements', fontsize=20)
    axs[1].set_xlabel('Month', fontsize=20)
    axs[1].legend(loc='best')
    #Save figure
    plt.tight_layout()
    fig.savefig('Images/year_month_label.png')

def words_counts(X, y, binary):
    '''
        Returns words and associated counts for either all real (binary=0) or all fake (binary=1) statements

        INPUT:
            - Array X
            - Array y
            - Int 0 or 1
        OUTPUT:
            - Array of words
            - Array of word counts
    '''
    vec = CountVectorizer(strip_accents='ascii', stop_words='english')
    vec_transform = vec.fit_transform(X[y==binary])
    counts = sum(vec_transform.toarray())
    words = vec.get_feature_names()
    return words, counts

def make_dictionary(arr1,arr2):
    '''
        Makes a dictionary out of two arrays.

        INPUT:
            - Array arr1
            - Array arr2
        OUTPUT:
            - Dictionary
    '''
    tuples = zip(arr1, arr2)
    result = list(tuples)
    dictionary = {}
    for tuple in result:
        dictionary[tuple[0]] = tuple[1]
    return dictionary

def make_wordcloud(word_arr1, freq_arr2):
    '''
        First makes a dictionary out of two arrays and then creates a wordcloud object from dictionary.

        INPUT:
            - Array word_arr1
            - Array freq_arr2
        OUTPUT:
            - Wordcloud object
    '''
    dictionary = make_dictionary(word_arr1, freq_arr2)
    wc = WordCloud(background_color="white",
               width=1000, height=1000,
               max_words=100, relative_scaling=0.25,
               normalize_plurals=False).generate_from_frequencies(dictionary)
    return wc

def plot_wordclouds(fake_words, fake_counts, real_words, real_counts):
    '''
        Plots two wordclouds- for words in fake statements and for words in real statements.

        INPUT:
            - Array fake_words
            - Array fake_counts
            - Array real_words
            - Array real_counts
        OUTPUT:
            - Matplotlib figure
    '''
    #Creates fake and real wordcloud objects
    wc_fake = make_wordcloud(fake_words, fake_counts)
    wc_real = make_wordcloud(real_words, real_counts)
    #Plots wordcloud objects
    fig, ax = plt.subplots(1,2, figsize=(20, 10))
    ax[0].set_title('Common Words in Lies', fontsize=30)
    ax[1].set_title('Common Words in Truths', fontsize=30)
    ax[0].axis('off')
    ax[1].axis('off')
    ax[0].imshow(wc_fake)
    ax[1].imshow(wc_real)
    #Saves figure
    fig.savefig('Images/wordclouds.png')