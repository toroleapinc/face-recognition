# Face Recognition with Eigenfaces

Comparing PCA (eigenface), LDA (fisherface), and Gabor filter features for face recognition on the ORL dataset. Uses SVM for classification.

Ran this for a pattern recognition course. The combined feature approach got best results at ~97.5% on ORL.

## Running

```
pip install -r requirements.txt
python train.py
python evaluate.py
```

Dataset: download ORL faces from https://cam-orl.co.uk/facedatabase.html and put in `data/orl_faces/`

## Results

| Method | Accuracy |
|--------|----------|
| Eigenface (PCA) | 93.3% |
| Fisherface (LDA) | 96.7% |
| Gabor + SVM | 95.0% |
| Combined | 97.5% |
