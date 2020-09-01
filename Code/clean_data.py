import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer

def get_data():
    '''
        Reads and cleans data.

        INPUT:
            - None
        OUTPUT:
            - Pandas dataframe
    '''
    #read data in from files
    data1 = pd.read_csv('Data/fnn_dev.csv')
    data2 = pd.read_csv('Data/fnn_test.csv')
    data3 = pd.read_csv('Data/fnn_train.csv')
    #concat data into one dataset
    data = pd.concat((data1,data2,data3)).reset_index().drop(['id','index'], axis=1)
    #replace real and fake with 0 and 1
    data['label_fnn'].replace(['real','fake'], [0,1], inplace=True)
    #add time columns
    data['date'] = data['date'].str.split('T',n=1)
    for i in range(len(data)):
        data['date'][i] = dt.strptime(data['date'][i][0], '%Y-%m-%d')
    data['year'] = pd.DatetimeIndex(data['date']).year
    data['month'] = pd.DatetimeIndex(data['date']).month
    #make column for number of characters per statement
    data['statement_len'] = 0
    for i in range(len(data)):
        data['statement_len'][i] = len(data['statement'][i])
    return data

def speaker_dataframe(data):
    '''
        Creates a dataframe for speakers.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Pandas dataframe
    '''
    #Creates dataframe from original dataframe grouped by speakers
    Speakers = pd.DataFrame(data.groupby(data['speaker'])['label_fnn'].count().sort_values(ascending=False)[:20]).reset_index()
    Speakers['statements'] = Speakers['label_fnn']
    Speakers = Speakers.drop('label_fnn', axis=1)
    #Adds columns for total lies, percent lies, and political party if applicable
    lies = []
    for i in Speakers['speaker']:
        lies.append(data[data['speaker'] == i]['label_fnn'].sum())
    Speakers['lies'] = lies
    Speakers['%lies'] = Speakers['lies']/Speakers['statements']
    Speakers['party'] = ['R','D','','','','D','R','R','R','','R','D','R','R','R','D','R','R','R','R']
    return Speakers

def split_X_y(data):
    '''
        Creates X column from statements and y column from labels.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Array X
            - Array y
    '''
    if 'label_fnn' in data.columns:
        y = data.pop('label_fnn')
        X = data['statement']
        return X, y

def avg_word_count(col):
    '''
        Finds the average number of words in a column of statements.

        INPUT:
            - Pandas series
        OUTPUT:
            - Float
    '''
    vec = CountVectorizer(strip_accents='ascii')
    vec_transform = vec.fit_transform(col)
    matrix = vec_transform.toarray()
    lst = []
    for i in matrix:
        lst.append(sum(i))
    return np.mean(lst)

def data_info(data):
    '''
        Calculates different measures in data and returns print statements.

        INPUT:
            - Pandas dataframe
        OUTPUT:
            - Print statements
    '''
    #Finds the number of real and fake statements
    num_real = len(data[data['label_fnn'] == 0])
    per_real = len(data[data['label_fnn'] == 0])/len(data)*100
    num_fake = len(data[data['label_fnn'] == 1])
    per_fake = len(data[data['label_fnn'] == 1])/len(data)*100
    #Finds the range of years in dataframe
    start_year = min(data['year'])
    end_year = max(data['year'])
    #Finds the number of unique speakers and the number of statements made by the top 20 speakers.
    speakers = data['speaker'].nunique()
    num_top20_state = sum(data.groupby(data['speaker'])['label_fnn'].count().sort_values(ascending=False)[:20])
    #Finds the average number of words and characters per statement
    avg_words = avg_word_count(data['statement'])
    avg_num_char = np.mean(data['statement_len'])

    print(f'Number of total data points: {len(data)}')
    print(f'Number of data labeled real: {num_real} ; percent real: {per_real:.1f}%')
    print(f'Number of data labeled fake: {num_fake} ; percent fake: {per_fake:.1f}%')
    print()
    print(f'Years data spans: {start_year} - {end_year}')
    print()
    print(f'Number of unique speakers: {speakers}')
    print(f'Number of statements made by top 20 speakers: {num_top20_state}')
    print()
    print(f'Average number of words per statement: {avg_words:.2f}')
    print(f'Average number of characters per statement: {avg_num_char:.2f}')