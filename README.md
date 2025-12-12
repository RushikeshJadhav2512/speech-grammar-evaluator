# speech-grammar-evaluator


🗣️ Grammar Scoring Engine from Voice Samples

🎯 Automatic Grammar Quality Scoring using Audio + Machine Learning

This project builds an end-to-end pipeline that evaluates spoken grammar quality from raw voice recordings. It extracts acoustic features, trains machine-learning models, and produces grammar-score predictions (0–5 range) as required in the competition.

⸻

📌 Features

✔ Load & process raw audio (.wav)
✔ Extract MFCCs, Mel-spectrogram, RMSE, ZCR, spectral contrast, tonnetz
✔ Automated feature engineering
✔ LightGBM model with 5-fold cross-validation
✔ Computes RMSE, Pearson correlation
✔ Generates visualizations & performance reports
✔ Produces submission.csv for final prediction
✔ Saves trained model and scaler for reuse

⸻

📁 Project Structure

Grammar Scoring Engine from Voice Samples/
│
├── grammar_scoring_engine.py     # Main training + inference script
├── dataset/
│   ├── audios/
│   │   ├── train/
│   │   └── test/
│   └── csvs/
│       ├── train.csv
│       └── test.csv
│
├── submission.csv                # Generated test predictions
├── train_oof_predictions.csv     # Out-of-fold predictions
├── scaler.joblib                 # Saved preprocessor
├── final_lgb_model.txt           # Saved model (LightGBM)
├── model_summary.txt             # Detailed performance report
└── README.md


⸻

🚀 How It Works

1️⃣ Load Data

Reads metadata from CSV and fetches audio files from train/test directories.

2️⃣ Extract Features

Using librosa, the system extracts:
	•	MFCC (mean + std)
	•	Mel-spectrogram features
	•	RMSE, ZCR
	•	Spectral contrast & Tonnetz
	•	Statistical waveform features

3️⃣ Train Model
	•	StandardScaler normalization
	•	LightGBM regression
	•	5-fold cross-validation
	•	Computes RMSE + Pearson correlation

4️⃣ Predict

Final model trained on full dataset → generates predictions for test set.

⸻

📊 Outputs Generated
	•	submission.csv — final predictions
	•	true_vs_pred.png — prediction scatter plot
	•	label_distribution.png
	•	pred_distribution.png
	•	distribution_comparison.png
	•	feature_importance.png

⸻

🛠 Requirements

Install dependencies:

pip install -r requirements.txt

Or manually:

pip install numpy pandas matplotlib seaborn librosa soundfile lightgbm joblib tqdm


⸻

▶️ Run the Engine

python grammar_scoring_engine.py


⸻

🔮 Future Improvements
	•	Use Wav2Vec2 / HuBERT / Whisper embeddings
	•	Add grammar error detection from text (ASR-based)
	•	Hyperparameter tuning with Optuna
	•	Audio augmentation (noise, pitch shift)
	•	Ensemble models
