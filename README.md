# Digital Audit Platform

## Overview

This repository contains the implementation of a digital audit platform designed to enhance the accuracy of audit scoring through advanced NLP and deep learning techniques. The project integrates BERT for embedding-based similarity measurements and supports various methods for evaluating user responses against reference answers.

## Features

- **Text Preprocessing**: Standardize and clean text data for accurate embedding and similarity calculations.
- **Embedding Techniques**: Utilize BERT for deep neural embeddings and similarity comparisons.
- **Similarity Measures**: Compute similarity scores using various techniques, including TF-IDF, LSA, and BERT.
- **Score Prediction**: Infer scores based on similarity scores from reference answers.
- **Simple Web Interface**: A basic HTML/CSS/JS frontend integrated with a Flask backend to interact with the audit platform.

## Architecture

- **Backend**: Python with Flask
  - Handles API requests, processes text data, and integrates with BERT for embedding.
- **Frontend**: HTML, CSS, JavaScript
  - Provides a simple interface for user interaction and displaying results.
- **Data Management**: CSV files for storing questions, reference answers, and scores.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/project-repository.git
