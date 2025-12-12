# Grammar Scoring Engine — Complete Implementation
# ---------------------------------------------------------------------------
# Purpose: End-to-end pipeline for the spoken-grammar scoring competition.
# Author: Grammar Scoring Engine Implementation

"""
CONTENTS
1. Setup & imports
2. Helper functions (audio loading, feature extraction)
3. Load CSVs and quick EDA
4. Feature extraction pipeline (audio features)
5. Model training (LightGBM) with cross-validation, compute RMSE & Pearson on training
6. Visualizations
7. Final model fit & test predictions -> submission.csv
8. Notes & next steps
"""

# 1) Setup & imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import soundfile as sf
import librosa
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Try to import LightGBM
try:
    import lightgbm as lgb
    print("LightGBM imported successfully")
except Exception as e:
    lgb = None
    print('LightGBM not found — please pip install lightgbm to use the example model')

# 2) Helper functions

def load_audio(path, sr=16000, mono=True, duration=None):
    """Load an audio file using librosa and return waveform and sample rate.
    duration in seconds (None => full file)
    """
    try:
        y, sr = librosa.load(path, sr=sr, mono=mono, duration=duration)
        return y, sr
    except Exception as e:
        print(f"Error loading audio file {path}: {e}")
        return np.zeros(16000), sr  # Return 1 second of silence as fallback

def extract_features(y, sr, n_mfcc=13, n_mels=128):
    """Extract a set of per-file aggregated audio features.
    Returns a 1D numpy array of features.
    """
    # Ensure numpy array
    y = np.asarray(y)
    if y.size == 0:
        # edge case: empty audio
        return np.zeros(100, dtype=float)

    # Basic energy / timing features
    rmse = librosa.feature.rms(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    # MFCCs: take mean and std for each coeff
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = mfcc.mean(axis=1)
    mfcc_stds = mfcc.std(axis=1)

    # Mel spectrogram summary
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_means = mel_db.mean(axis=1)
    mel_stds = mel_db.std(axis=1)

    # Spectral contrast
    try:
        contrast = librosa.feature.spectral_contrast(S=None, y=y, sr=sr, n_bands=6)
        contrast_means = contrast.mean(axis=1)
        contrast_stds = contrast.std(axis=1)
    except Exception:
        contrast_means = np.zeros(7)  # 7 bands by default
        contrast_stds = np.zeros(7)

    # Tonnetz (requires harmonic)
    try:
        y_harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
        tonnetz_means = tonnetz.mean(axis=1)
        tonnetz_stds = tonnetz.std(axis=1)
    except Exception:
        tonnetz_means = np.zeros(6)
        tonnetz_stds = np.zeros(6)

    # Statistical features
    features = np.hstack([
        np.array([rmse.mean(), rmse.std()]),
        np.array([zcr.mean(), zcr.std()]),
        mfcc_means, mfcc_stds,
        mel_means[:10], mel_stds[:10],  # limit mel features to keep dimension reasonable
        contrast_means, contrast_stds,
        tonnetz_means, tonnetz_stds,
        np.array([y.mean(), y.std(), np.percentile(y, 1), np.percentile(y, 99)])
    ])

    return features

# 3) Paths and CSV loading - CORRECTED PATHS
TRAIN_CSV = 'dataset/csvs/train.csv'
TEST_CSV = 'dataset/csvs/test.csv'
TRAIN_AUDIO_DIR = 'dataset/audios/train'  # folder containing wav files
TEST_AUDIO_DIR = 'dataset/audios/test'

# Basic safety: check files exist
if not os.path.exists(TRAIN_CSV):
    print(f'Warning: {TRAIN_CSV} not found.')
if not os.path.exists(TEST_CSV):
    print(f'Warning: {TEST_CSV} not found.')

# Load CSVs
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print('Train samples:', len(train_df))
print('Test samples:', len(test_df))
print('Train CSV columns:', train_df.columns.tolist())
print('Test CSV columns:', test_df.columns.tolist())

# Quick EDA on labels (if present)
if 'label' in train_df.columns:
    print('\nLabel distribution:')
    print(train_df['label'].describe())
    plt.figure(figsize=(6,3))
    sns.histplot(train_df['label'], bins=20, kde=True)
    plt.title('Train label distribution')
    plt.xlabel('Grammar Score')
    plt.ylabel('Count')
    plt.savefig('label_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

# 4) Feature extraction for all audio files (this may take time)

def build_feature_matrix(df, audio_column='filename', audio_dir=TRAIN_AUDIO_DIR, sr=16000):
    features = []
    filenames = []
    missing_files = []
    
    for fname in tqdm(df[audio_column].values, desc='Extracting features'):
        # Handle different audio file extensions if needed
        fp = os.path.join(audio_dir, fname)
        if not fp.endswith('.wav'):
            fp += '.wav'
            
        if not os.path.exists(fp):
            # Try without .wav extension
            fp = os.path.join(audio_dir, fname)
            if not os.path.exists(fp):
                print(f'Missing file: {fname}')
                missing_files.append(fname)
                feat = np.zeros(100)
            else:
                try:
                    y, sr = load_audio(fp, sr=sr)
                    feat = extract_features(y, sr)
                except Exception as e:
                    print(f'Error reading {fp}, using zeros: {e}')
                    feat = np.zeros(100)
        else:
            try:
                y, sr = load_audio(fp, sr=sr)
                feat = extract_features(y, sr)
            except Exception as e:
                print(f'Error reading {fp}, using zeros: {e}')
                feat = np.zeros(100)
                
        features.append(feat)
        filenames.append(fname)
    
    print(f'Missing files: {len(missing_files)}')
    X = np.vstack(features)
    return X, filenames

print('Building train feature matrix...')
X_train_raw, train_files = build_feature_matrix(train_df, audio_column='filename', audio_dir=TRAIN_AUDIO_DIR)

print('Building test feature matrix...')
X_test_raw, test_files = build_feature_matrix(test_df, audio_column='filename', audio_dir=TEST_AUDIO_DIR)

print('Raw feature shapes:', X_train_raw.shape, X_test_raw.shape)

# 5) Preprocessing: scaling + optional PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Optional PCA to reduce dimensionality (uncomment if you want to use)
# pca = PCA(n_components=60)
# X_train = pca.fit_transform(X_train_scaled)
# X_test = pca.transform(X_test_scaled)
X_train = X_train_scaled
X_test = X_test_scaled

# 6) Model training with cross-validation & compute RMSE on training set

y_train = train_df['label'].values

# K-Fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
pearson_scores = []
preds_oof = np.zeros(len(X_train))

if lgb is None:
    print('LightGBM not available. Using Linear Regression as fallback.')
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import cross_val_score
    
    model = LinearRegression()
    rmse_cv = np.sqrt(-cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    print(f'Cross-validation RMSE: {rmse_cv.mean():.4f} (+/- {rmse_cv.std() * 2:.4f})')
    
    model.fit(X_train, y_train)
    test_preds = model.predict(X_test)
    
else:
    print('Training LightGBM model with cross-validation...')
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f'Processing fold {fold+1}/5...')
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'verbosity': -1,
            'seed': 42
        }


        model = lgb.train(params, train_data, num_boost_round=1000,
                          valid_sets=[train_data, valid_data],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])

        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        preds_oof[val_idx] = val_pred


        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        try:
            pearson = pearsonr(y_val, val_pred)[0]
        except Exception:
            pearson = np.nan

        rmse_scores.append(rmse)
        pearson_scores.append(pearson)
        print(f'Fold {fold+1} RMSE: {rmse:.4f}  Pearson: {pearson:.4f}')

    # Overall training RMSE (OOF)

    overall_rmse = np.sqrt(mean_squared_error(y_train, preds_oof))
    try:
        overall_pearson = pearsonr(y_train, preds_oof)[0]
    except Exception:
        overall_pearson = np.nan

    print('\nOOF TRAIN RMSE:', overall_rmse)
    print('OOF TRAIN Pearson:', overall_pearson)

    # Save predictions & metrics to disk
    pd.DataFrame({'filename': train_files, 'oof_pred': preds_oof, 'label': y_train}).to_csv('train_oof_predictions.csv', index=False)

    # 7) Fit final model on full training data and predict test set
    final_train_data = lgb.Dataset(X_train, label=y_train)
    final_params = params
    final_model = lgb.train(final_params, final_train_data, num_boost_round=model.best_iteration)

    test_preds = final_model.predict(X_test, num_iteration=final_model.best_iteration)

# Clip to [0,5] as target domain
test_preds = np.clip(test_preds, 0.0, 5.0)

submission = pd.DataFrame({'filename': test_files, 'label': test_preds})
submission.to_csv('submission.csv', index=False)
print('\nSaved submission.csv with', len(submission), 'rows.')

# 8) Visualizations
plt.figure(figsize=(6,4))
sns.scatterplot(x=y_train, y=preds_oof)
plt.xlabel('True label')
plt.ylabel('OOF prediction')
plt.title('True vs OOF Prediction')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', alpha=0.5)
plt.savefig('true_vs_pred.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(preds_oof, bins=30)
plt.title('Distribution of OOF predictions')
plt.xlabel('Predicted Score')
plt.ylabel('Count')
plt.savefig('pred_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(6,4))
sns.histplot(test_preds, bins=30, alpha=0.7, label='Test predictions')
sns.histplot(y_train, bins=30, alpha=0.7, label='Training labels')
plt.title('Distribution Comparison')
plt.xlabel('Score')
plt.ylabel('Count')
plt.legend()
plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 9) Feature importance (if using LightGBM)
if lgb is not None:
    feature_importance = pd.DataFrame({
        'feature': [f'feature_{i}' for i in range(X_train.shape[1])],
        'importance': final_model.feature_importance()
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(8,6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()

# 10) Model performance summary
print("\n" + "="*50)
print("MODEL PERFORMANCE SUMMARY")
print("="*50)
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Feature dimensions: {X_train.shape[1]}")
if lgb is not None:
    print(f"Cross-validation RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(f"Cross-validation Pearson: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}")
    print(f"OOF RMSE: {overall_rmse:.4f}")
    print(f"OOF Pearson: {overall_pearson:.4f}")
else:
    print(f"Cross-validation RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}")

print(f"Test predictions range: {test_preds.min():.2f} - {test_preds.max():.2f}")
print(f"Training labels range: {y_train.min():.2f} - {y_train.max():.2f}")

# 11) Save preprocessing objects for later use
import joblib
joblib.dump(scaler, 'scaler.joblib')
if lgb is not None:
    final_model.save_model('final_lgb_model.txt')

# 12) Notes and next steps
notes = '''
Next steps & improvements you can try:
- Use a stronger pre-trained audio model (Wav2Vec2, HuBERT, or OpenL3) to extract embeddings instead of hand-crafted features.
- Run an ASR model (e.g., Whisper) to get transcripts and compute textual grammar/error features: grammar error counts, perplexity, POS-tag patterns, error detectors.
- Use stacking/ensembling: combine audio-embedding models + text-based models + hand-crafted features.
- Tune LightGBM hyperparameters with Optuna or GridSearchCV.
- Use sample-wise data augmentation (time-stretch, pitch shift, noise) and mixup to improve robustness.
- Consider ordinal regression or customized loss that respects the 0-5 continuous scale.
- Add confidence calibration and per-speaker normalization if speaker IDs are available.
'''

print(notes)

# Create a summary report
with open('model_summary.txt', 'w') as f:
    f.write("Grammar Scoring Engine - Model Summary\n")
    f.write("="*50 + "\n\n")
    f.write(f"Training samples: {len(train_df)}\n")
    f.write(f"Test samples: {len(test_df)}\n")
    f.write(f"Feature dimensions: {X_train.shape[1]}\n")
    if lgb is not None:
        f.write(f"Cross-validation RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}\n")
        f.write(f"Cross-validation Pearson: {np.mean(pearson_scores):.4f} ± {np.std(pearson_scores):.4f}\n")
        f.write(f"OOF RMSE: {overall_rmse:.4f}\n")
        f.write(f"OOF Pearson: {overall_pearson:.4f}\n")
    else:
        f.write(f"Cross-validation RMSE: {rmse_cv.mean():.4f} ± {rmse_cv.std():.4f}\n")
    f.write(f"Test predictions range: {test_preds.min():.2f} - {test_preds.max():.2f}\n")
    f.write(f"Training labels range: {y_train.min():.2f} - {y_train.max():.2f}\n")
    f.write("\nGenerated files:\n")
    f.write("- submission.csv: Test set predictions\n")
    f.write("- train_oof_predictions.csv: Out-of-fold training predictions\n")
    f.write("- model_summary.txt: This summary report\n")
    f.write("- scaler.joblib: Trained feature scaler\n")
    if lgb is not None:
        f.write("- final_lgb_model.txt: Trained LightGBM model\n")
    f.write("- Various visualization PNG files\n")

print("\nGrammar Scoring Engine completed successfully!")
print("Check the generated files: submission.csv, visualizations, and model summary.")

