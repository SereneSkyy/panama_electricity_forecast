# Panama Energy AI — RNN vs. LSTM Demand Forecasting

```bash
# 1. Install dependencies (from the project root)
pip install -r requirements.txt

# 2. (Re-)train both models -- writes models/*.pth and models/*.pkl
python src/train.py

# 3. Check both models' metrics on the test set
python src/evaluate.py

# 4. Start the backend API
uvicorn main:app --reload
# -> running at http://127.0.0.1:8000

# 5. Open the dashboard
# Just open dashboard/index.html directly in a browser, or serve it:
python -m http.server 5500 --directory dashboard
# -> open http://127.0.0.1:5500
```

The dashboard fetches a random 24-hour window from the backend, shows the actual demand curve plus
the model's next-hour prediction, and lets you switch between RNN and LSTM on the _same_ window for
a fair side-by-side comparison — click "Fetch new data window" for a new sample.
