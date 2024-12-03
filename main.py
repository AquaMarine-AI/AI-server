from flask import Flask, request, jsonify
import os
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import MinMaxScaler
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

app = Flask(__name__)

# 파일 경로 정의
MODEL_PATH = "model_checkpoint.pth"
SCALER_PATH = "scaler.pkl"
DATASET_PATH = "timeseries_dataset.pkl"

with open(DATASET_PATH, "rb") as f:
    dataset = joblib.load(f)

# 속성 출력
print("Dataset attributes:")
print(f"Static categoricals: {dataset.static_categoricals}")
print(f"Time-varying known reals: {dataset.time_varying_known_reals}")
print(f"Time-varying unknown reals: {dataset.time_varying_unknown_reals}")
print(f"Max encoder length: {dataset.max_encoder_length}")
print(f"Max prediction length: {dataset.max_prediction_length}")
print("Dataset full attributes:")
print(dataset.__dict__)


# 디렉토리 설정
UPLOAD_DIR = "uploads/uploaded_files"
PROCESSED_DIR = "uploads/processed_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# CSV 처리 함수
def process_csv(file):
    file_name = file.filename
    file_location = os.path.join(UPLOAD_DIR, file_name)
    file.save(file_location)
    processed_file_location = os.path.join(PROCESSED_DIR, f"processed_{file_name}")
    processed_data = pd.read_csv(file_location)
    processed_data.to_csv(processed_file_location, index=False)
    return processed_file_location

# 모델 및 데이터셋 복원 함수
def load_model_and_scaler(model_path, scaler_path, dataset_path):
    try:
        print("[INFO] Loading TimeSeriesDataSet...")
        with open(dataset_path, "rb") as f:
            dataset = joblib.load(f)
        
        # 속성 기본값 설정
        dataset.static_reals = dataset.static_reals or []
        dataset.static_categoricals = dataset.static_categoricals or []
        dataset.time_varying_known_categoricals = dataset.time_varying_known_categoricals or []
        dataset.time_varying_unknown_categoricals = dataset.time_varying_unknown_categoricals or []
        dataset.allow_missing_timesteps = dataset.allow_missing_timesteps or False

        print("[INFO] Dataset attributes after setting defaults:")
        print(f"Static categoricals: {dataset.static_categoricals}")
        print(f"Static reals: {dataset.static_reals}")
        print(f"Time-varying known categoricals: {dataset.time_varying_known_categoricals}")
        print(f"Time-varying unknown categoricals: {dataset.time_varying_unknown_categoricals}")
        print(f"Max encoder length: {dataset.max_encoder_length}")
        print(f"Max prediction length: {dataset.max_prediction_length}")

        print("[INFO] Loading TemporalFusionTransformer model...")
        state_dict = torch.load(model_path)
        if "model_state_dict" not in state_dict:
            raise KeyError("Key 'model_state_dict' not found in state_dict. Check the saved model checkpoint.")

        model = TemporalFusionTransformer.from_dataset(dataset)
        model.load_state_dict(state_dict["model_state_dict"], strict=False)
        print("[INFO] Model loaded successfully.")

        print("[INFO] Loading scaler...")
        scaler = joblib.load(scaler_path)
        if scaler is None:
            raise ValueError("Scaler is None. Check the saved scaler file.")
        print("[INFO] Scaler loaded successfully.")

        return model, scaler, dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load model or scaler: {e}")



# 예측 함수
def rolling_predict_until_condition(model, data, max_encoder_length, scaler, condition_value, tolerance, device="cuda"):
    print("[INFO] Starting rolling prediction...")
    model = model.to(device)
    model.eval()

    if len(data) < max_encoder_length:
        raise ValueError("Input data length is shorter than max_encoder_length.")

    input_data = data.iloc[:max_encoder_length].copy()
    predictions = []

    if "group" not in input_data.columns:
        input_data["group"] = "series"

    step = 0
    while True:
        try:
            rolling_dataset = TimeSeriesDataSet(
                data=input_data,
                time_idx="time_idx",
                target="feed_pressure",
                group_ids=["group"],
                max_encoder_length=max_encoder_length,
                max_prediction_length=1,
                static_categoricals=[],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=["feed_pressure"],
                target_normalizer=GroupNormalizer(transformation="relu"),
                min_encoder_length=1,
                allow_missing_timesteps=True,
            )
            rolling_dataloader = rolling_dataset.to_dataloader(train=False, batch_size=1)

            with torch.no_grad():
                input_batch = next(iter(rolling_dataloader))
                input_batch = {key: value.to(device) for key, value in input_batch[0].items()}
                prediction = model(input_batch)["prediction"].cpu().numpy().squeeze()

            prediction_actual = scaler.inverse_transform([[prediction]]).flatten()[0]
            predictions.append(prediction_actual)

            print(f"[Step {step}] Predicted value: {prediction_actual}")
            if abs(prediction_actual - condition_value) <= tolerance:
                print(f"[INFO] Condition met at step {step}.")
                break

            new_time_idx = input_data["time_idx"].iloc[-1] + 1
            new_row = {
                "time_idx": new_time_idx,
                "feed_pressure": float(prediction),
                "group": "series",
            }
            input_data = pd.concat([input_data.iloc[1:], pd.DataFrame([new_row])], ignore_index=True)
            step += 1

        except Exception as e:
            print(f"[ERROR] Error during prediction step {step}: {e}")
            break

    return predictions


@app.route("/api/v1/predict", methods=["POST"])
def upload_csv():
    try:
        if "file" not in request.files:
            return jsonify({"message": "No file part in request"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"message": "No selected file"}), 400

        print("[INFO] Processing uploaded file...")
        processed_file_location = process_csv(file)
        df = pd.read_csv(processed_file_location)
        print("[INFO] File processed successfully.")

        # 모델 및 스케일러 로드
        model, scaler, dataset = load_model_and_scaler(MODEL_PATH, SCALER_PATH, DATASET_PATH)

        # 데이터 전처리
        df["feed_pressure"] = scaler.transform(df[["feed_pressure"]])
        df["time_idx"] = range(len(df))
        if "group" not in df.columns:
            df["group"] = "series"

        # max_encoder_length 기본값 설정
        max_encoder_length = dataset.max_encoder_length if dataset.max_encoder_length else 12

        # 예측 수행
        predictions = rolling_predict_until_condition(
            model=model,
            data=df,
            max_encoder_length=max_encoder_length,
            scaler=scaler,
            condition_value=17,
            tolerance=0.1,
            device="cuda"
        )

        return jsonify({"predictions": predictions}), 200

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
