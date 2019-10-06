#!/usr/bin/env python
# coding: utf-8

import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ## Preprocess functions
label_vals = {'EOS': 1, 'NEOS': 0}
inverse_label_vals = {1: 'EOS', 0: 'NEOS'}


def preprocess(raw):
    
    # Split data into separate columns
    raw['pos'] = raw['data'].str.split().str[0]
    raw['word'] = raw['data'].str.split().str[1]
    raw['type'] = raw['data'].str.split().str[2]
    raw.index = raw['pos'].astype(int)
    
    # Drop, rename columns
    raw = raw.drop(['data', 'pos'], axis=1)
    raw = raw.rename(columns={'word': 'L', 'type': 'label'})

    # Get next word in the same row to ease computation later
    raw['R'] = raw['L'].shift(-1)
    raw['R'] = raw['R'].fillna('EndOfDocument')

    # Choose only those periods which are of the form 'L. R'
    raw = raw[raw['label'].isin(['EOS', 'NEOS'])]
    
    # Remove the attached period to the words
    raw['L'] = raw['L'].str.strip('.')
    raw['R'] = raw['R'].str.strip('.')
    return raw


def string_encoder(df):
    
    # Encode L and R strings with a unique number for each word
    corpus = list(set(df['L'].tolist() + df['R'].tolist()))
    wordmap_inverse = pd.Series(data=corpus).to_dict()
    wordmap = {wordmap_inverse[k]: k for k in wordmap_inverse}
    return wordmap


# Generate 5 core features mentioned in assignment

def generate_core_features(df, wordmap):
    df['L_encoded'] = df['L'].map(wordmap).fillna(-1).astype(int)
    df['R_encoded'] = df['R'].map(wordmap).fillna(-1).astype(int)
    
    df['L<3'] = df['L'].str.len() < 3
    df['L_cap'] = df['L'].str.istitle()
    df['R_cap'] = df['R'].str.istitle()
    return df


# Additional features: 
# 
# 1. L contains period
# 2. R is comma
# 3. Length of R


def generate_additional_features(df):
    df['L_contains_period'] = df['L'].str.contains('\.')
    df['R_comma'] = df['R'] == ','
    df['R_length'] = df['R'].str.len()
    return df


def create_train_dataset(raw):
    df = preprocess(raw)
    wordmap = string_encoder(df)
    df = generate_core_features(df, wordmap)
    df = generate_additional_features(df)
    df['label_bin'] = df['label'].map(label_vals)
    return df, wordmap


def create_test_dataset(raw, wordmap):
    df = preprocess(raw)
    df = generate_core_features(df, wordmap)
    df = generate_additional_features(df)
    
    label_vals = {'EOS': 1, 'NEOS': 0}
    df['label_bin'] = df['label'].map(label_vals)
    return df


if __name__ == '__main__':
    # ## Main script
    train_file = sys.argv[1]
    test_file = sys.argv[2]

    # Read files as pandas dataframes
    raw_train = pd.read_csv(train_file, sep='\t', names=['data'])
    raw_test = pd.read_csv(test_file, sep='\t', names=['data'])

    train, wordmap = create_train_dataset(raw_train)
    test = create_test_dataset(raw_test, wordmap)

    # Train a Decision Tree classifier on training data

    features = ['L_encoded', 'R_encoded', 'L<3', 'L_cap', 'R_cap', 'L_contains_period', 'R_comma', 'R_length']
    # features = ['L_encoded', 'R_encoded', 'L<3', 'L_cap', 'R_cap']
    # features = ['L_contains_period', 'R_comma', 'R_length']
    target = ['label_bin']

    X_train = train[features].values
    y_train = train[target].values

    X_test = test[features].values
    y_test = test[target].values

    clf = DecisionTreeClassifier(max_depth=6)
    clf.fit(X_train, y_train)

    # Evaluation

    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)

    print('Training accuracy: {}'.format(accuracy_score(y_train, pred_train) * 100))
    print('Test accuracy: {}'.format(accuracy_score(y_test, pred_test) * 100))

    # Write test file in prescribed format
    test['prediction'] = pred_test
    test['prediction'] = test['prediction'].map(inverse_label_vals)
    test = test.reset_index()
    test[['pos', 'L', 'prediction']].to_csv('SBD.test.out', index=False)

    # ft_importance = pd.Series(data=clf.feature_importances_, index=features)
    # ft_importance.sort_values(ascending=False)
