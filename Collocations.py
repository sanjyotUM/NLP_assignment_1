#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
import numpy as np


# Parse the raw text into a dataframe
def load_file(filename):
    with open(filename, 'r') as f:
        raw_text = f.read()

    text_list = raw_text.strip().split(' ')  # Split single string into words

    # Create a dataframe with unigram and bigrams in text
    ngram = (
        pd.DataFrame(data=text_list, columns=['unigram'])
        .assign(unigram = lambda x: x['unigram'].str.strip())  # Remove whitespace and newline characters attached to words
        .loc[lambda x: ~x['unigram'].isin([',', '.', "'", '"', '`', '-', ':'])]  # Remove lone punctuation marks
        .assign(second_word = lambda x: x['unigram'].shift(-1).fillna('EndOfDocument'))  # Get the next word in the same row
        .assign(bigram = lambda x: x['unigram'].str.cat(x['second_word'], sep='::'))  # Join first and second word to make bigram
        .drop(['second_word'], axis=1)
    )
    return ngram


# Preprocess the counts to generate helping terms in chi square and PMI calculation
def preprocess(unicount, bicount):
    df = bicount.copy()
    
    # Separate bigram
    df['first'] = df['bigram'].str.split('::').str[0]
    df['second'] = df['bigram'].str.split('::').str[1]
    df = df.drop(['bigram'], axis=1)

    df = (
        df
        .merge(unicount, how='left', left_on='first', right_on='unigram')  # Get first word count
        .drop(['unigram'], axis=1)
        .rename(columns={'unicount': 'first_unigram_count'})
        .merge(unicount, how='left', left_on='second', right_on='unigram')  # Get second word count
        .drop(['unigram'], axis=1)
        .rename(columns={'unicount': 'second_unigram_count'})
        .assign(
            first_and_second=lambda x: x['bicount'],  # First and second word bigram count
            first_not_second=lambda x: x['first_unigram_count'] - x['bicount'],  # First word but not second word bigram count
            not_first_second=lambda x: x['second_unigram_count'] - x['bicount'],  # Not first word but second word bigram count
            not_first_not_second=lambda x: x['bicount'].sum() - x['first_unigram_count'] - x['second_unigram_count'] + x['bicount'],  # Not first and not second word bigram count
            total_unigram=unicount['unicount'].sum(),
            total_bigram=unicount['unicount'].sum() - 1
        )
    )
    return df


def get_chi_square_score(unicount, bicount):
        
    df = preprocess(unicount, bicount)

    # Generate observed and expected frequencies, and calculate the respective terms
    df['term11_obs'] = df['first_and_second']
    df['term11_exp'] = ((df['first_unigram_count']/df['total_unigram']) * (df['second_unigram_count']/df['total_unigram']) * df['total_bigram'])
    df['term11'] = np.square(df['term11_obs'] - df['term11_exp'])/df['term11_exp']
    df = df.drop(['term11_obs', 'term11_exp'], axis=1)

    df['term12_obs'] = df['not_first_second']
    df['term12_exp'] = ((1 - df['first_unigram_count']/df['total_unigram']) * (df['second_unigram_count']/df['total_unigram']) * df['total_bigram'])
    df['term12'] = np.square(df['term12_obs'] - df['term12_exp'])/df['term12_exp']
    df = df.drop(['term12_obs', 'term12_exp'], axis=1)

    df['term21_obs'] = df['first_not_second']
    df['term21_exp'] = ((df['first_unigram_count']/df['total_unigram']) * (1 - df['second_unigram_count']/df['total_unigram']) * df['total_bigram'])
    df['term21'] = np.square(df['term21_obs'] - df['term21_exp'])/df['term21_exp']
    df = df.drop(['term21_obs', 'term21_exp'], axis=1)

    df['term22_obs'] = df['not_first_not_second']
    df['term22_exp'] = ((1 - df['first_unigram_count']/df['total_unigram']) * (1 - df['second_unigram_count']/df['total_unigram']) * df['total_bigram'])
    df['term22'] = np.square(df['term22_obs'] - df['term22_exp'])/df['term22_exp']
    df = df.drop(['term22_obs', 'term22_exp'], axis=1)

    df['chi-square'] = df['term11'] + df['term12'] + df['term21'] + df['term22']
    return df


def get_pmi_score(unicount, bicount):
    df = preprocess(unicount, bicount)
    
    # Get probabilities and calculate PMI
    df['P_w1w2'] = (df['first_and_second']/df['total_bigram'])
    df['P_w1'] = (df['first_unigram_count']/df['total_unigram'])
    df['P_w2'] = (df['second_unigram_count']/df['total_unigram'])
    df['PMI'] = np.log2(df['P_w1w2']/(df['P_w1'] * df['P_w2']))
    return df


def get_asked_score(unicount, bicount, measure):
    if measure == 'chi-square':
        return get_chi_square_score(unicount, bicount)
    elif measure == 'PMI':
        return get_pmi_score(unicount, bicount)
    else:
        print("Invalid option for measure, choose 'chi-square' or 'PMI'")


# Main Script
if __name__ == '__main__':
    filename = sys.argv[1]
    measure = sys.argv[2]

    ngram = load_file(filename)

    # Get counts of unigrams and bigrams
    unigram_count = (
        ngram
        .groupby(by=['unigram'])
        .agg({'unigram': 'count'})
        .rename(columns={'unigram': 'unicount'})
        .reset_index()
        .sort_values(by='unicount', ascending=False)
    )

    bigram_count = (
        ngram.groupby(by=['bigram'])
        .agg({'bigram': 'count'})
        .rename(columns={'bigram': 'bicount'})
        .reset_index()
        .sort_values(by='bicount', ascending=False)
    )

    # Calculate the asked score
    df = get_asked_score(unigram_count, bigram_count, measure)

    print('{} score:\n'.format(measure))
    print(df.set_index(['first', 'second'])[measure].sort_values(ascending=False).head(20))
