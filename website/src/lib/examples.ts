export const quickStart = `from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from omnitabular import OmniTabular

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
model = OmniTabular(
    task="classification", models=["lgb", "rf", "linear"]
)
model.fit(X_train, y_train)
print(model.leaderboard())
print(model.predict(X_test[:5]))`;

export const localSetup = `git clone https://github.com/kaarse/omnitabular.git
cd omnitabular
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,api]"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8080`;
