# movie-recommendation-system
---

## 📖 Overview

This project builds a recommendation system that predicts movie ratings based on user and movie interactions. Using embedding layers for both users and movies, the model learns latent factors and biases, combines them with a dot product, and adjusts predictions within the original rating range.

---

## 📊 Dataset

* **Source:** [MovieLens Latest Small Dataset](https://grouplens.org/datasets/movielens/)
* Contains:

  * `movies.csv` — movie information (titles & genres)
  * `ratings.csv` — user-movie ratings

---

## 📦 Dependencies

* Python 3.x
* pandas
* numpy
* scikit-learn
* TensorFlow / Keras
* matplotlib (optional for visualization)
* graphviz (for model plot)

Install via:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib
```

---

## 📌 Features

* Download and preprocess the MovieLens dataset.
* Select top `k` active users and popular movies.
* Encode user and movie IDs.
* Build a Keras model using:

  * **Embedding layers** for user and movie latent factors
  * **Dot product** for interaction
  * **Bias terms** for users and movies
  * **Sigmoid activation** scaled back to rating range
* Train and evaluate using **RMSE**
* Visualize the model structure with `plot_model`

---

## 📈 Model Architecture

* User Embedding (size=50)
* Movie Embedding (size=50)
* Dot product of embeddings
* Add user and movie bias
* Sigmoid activation
* Scale to original rating range (0.5 to 5.0)
* Optimizer: Adam
* Loss: Mean Squared Error
* Metric: Root Mean Squared Error (RMSE)

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/movielens-collaborative-filtering.git
cd movielens-collaborative-filtering
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook or Python script:

```bash
python movielens_recommender.py
```

4. Model summary and plot will be generated as `model.png`

---

## 📊 Sample Result

After training for 8 epochs, the model achieves a test RMSE around **0.7-0.9**, depending on random seed and hyperparameters.

---


## ✨ Acknowledgments

* [MovieLens](https://grouplens.org/datasets/movielens/)
* TensorFlow Keras documentation
* Collaborative filtering concepts

---

