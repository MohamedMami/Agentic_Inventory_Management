import logging
import json
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from statsmodels.tsa.arima.model import ARIMA
from groq import Groq  
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class ForecastAgent:
    MODELS_DIR = "models"
    META_PATH = os.path.join(MODELS_DIR, "metadata.json")
    RETRAIN_THRESHOLD_DAYS = 30

    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0.1):
        self.groq_api = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_name = model_name
        self.temperature = temperature
        os.makedirs(self.MODELS_DIR, exist_ok=True)

    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Extract parameters using simple regex."""
        params = {
            "product_id": None,
            "product_name": None,
            "category": None,
            "horizon": 30,
            "p": 1,
            "d": 1,
            "q": 1,
            "confidence_interval": 0.95
        }
        # Example regex-based parsing (simplified)
        import re
        product_name_match = re.search(r"for\s+([\w\s]+)\s+", query)
        if product_name_match:
            params["product_name"] = product_name_match.group(1)
        
        horizon_match = re.search(r"for\s+the\s+next\s+(\d+)\s+days", query)
        if horizon_match:
            params["horizon"] = int(horizon_match.group(1))
        
        # Add more regex patterns for p, d, q, category, etc.
        return params

    def _generate_insights(self, product_name: str, forecast_data: List[Dict], statistics: Dict) -> str:
        """Generate insights without LLM (simple heuristic-based)."""
        mean = statistics.get("mean", 0)
        std = statistics.get("std", 0)
        aic = statistics.get("aic", 0)
        insights = (
            f"Forecast for {product_name}: "
            f"Average demand is {mean:.2f} units/day with a standard deviation of {std:.2f}. "
            f"ARIMA model AIC: {aic}. "
            f"Confidence intervals suggest moderate uncertainty."
        )
        return insights

    def _fetch_historical_data(self, session: Session, params: Dict) -> pd.DataFrame:
        """Same as before but without Pydantic."""
        sql = "SELECT sale_date::date AS date, SUM(quantity) AS sales, product_id, product_name, category FROM sales"
        filters = []
        query_params = {}
        if params.get("product_name"):
            filters.append("product_name ILIKE :pname")
            query_params["pname"] = f"%{params['product_name']}%"
        if params.get("category"):
            filters.append("category = :cat")
            query_params["cat"] = params["category"]
        if filters:
            sql += " WHERE " + " AND ".join(filters)
        sql += " GROUP BY date, product_id, product_name, category ORDER BY date"
        result = session.execute(text(sql), query_params)
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").asfreq("D").fillna(0).reset_index()
        return df

    def _generate_forecast(self, df: pd.DataFrame, params: Dict) -> Dict:
        """Same ARIMA logic but without Pydantic."""
        ts = df.set_index("date")["sales"]
        key = params.get("product_id") or df["product_id"].iloc[0]
        model = self._load_or_update_model(ts, key, params)
        fc = model.get_forecast(steps=params["horizon"])
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=1 - params["confidence_interval"])
        dates = pd.date_range(ts.index.max() + timedelta(days=1), periods=params["horizon"])
        forecast_values = [
            {
                "date": d.strftime("%Y-%m-%d"),
                "forecast": float(mean[i]),
                "lower": float(ci.iloc[i, 0]),
                "upper": float(ci.iloc[i, 1])
            }
            for i, d in enumerate(dates)
        ]
        stats = {
            "arima_order": f"({params['p']},{params['d']},{params['q']})",
            "aic": model.aic,
            "mean": mean.mean(),
            "std": mean.std()
        }
        prod_info = {
            "id": df["product_id"].iloc[0],
            "name": df["product_name"].iloc[0],
            "category": df["category"].iloc[0]
        }
        viz = self._create_visualization(ts, mean, ci, dates, prod_info["name"])
        return {
            "product_info": prod_info,
            "forecast_values": forecast_values,
            "statistics": stats,
            "visualization_base64": viz,
            "insights": self._generate_insights(prod_info["name"], forecast_values, stats)
        }

    def _load_or_update_model(self, ts: pd.Series, key: str, params: Dict):
        """Same as before but without Pydantic."""
        meta = self._load_metadata()
        last_date = ts.index.max().date()
        if meta.get(key):
            trained_date = datetime.fromisoformat(meta[key]["last_trained"]).date()
            if (last_date - trained_date).days <= self.RETRAIN_THRESHOLD_DAYS:
                return joblib.load(meta[key]["model_path"])
        # Retrain
        model, meta_entry = self._train_model(ts, key, params)
        meta[key] = meta_entry
        self._save_metadata(meta)
        return model

    def _train_model(self, ts: pd.Series, key: str, params: Dict):
        model = ARIMA(ts, order=(params["p"], params["d"], params["q"])).fit()
        now = ts.index.max().date().isoformat()
        model_path = os.path.join(
            self.MODELS_DIR,
            f"{key}_{params['p']}{params['d']}{params['q']}.pkl"
        )
        joblib.dump(model, model_path)
        return model, {"last_trained": now, "model_path": model_path}

    def _load_metadata(self) -> Dict:
        if os.path.exists(self.META_PATH):
            return json.load(open(self.META_PATH))
        return {}

    def _save_metadata(self, meta: Dict):
        json.dump(meta, open(self.META_PATH, "w"), indent=2)

    def _create_visualization(self, hist, mean, ci, dates, name):
        """Same as before."""
        plt.figure(figsize=(10,6))
        plt.plot(hist.index, hist.values, label="History")
        plt.plot(dates, mean, label="Forecast")
        plt.fill_between(dates, ci.iloc[:,0], ci.iloc[:,1], alpha=0.3)
        plt.title(f"Forecast for {name}")
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    def process_query(self, query: str, session: Session) -> Dict[str, Any]:
        try:
            params = self._parse_query(query)
            df = self._fetch_historical_data(session, params)
            if df.empty:
                return {"error": "No data found.", "response": "No historical data available."}
            result = self._generate_forecast(df, params)
            return {"response": "Forecast generated.", "data": result}
        except Exception as e:
            logger.error(f"Forecast error: {str(e)}")
            return {"error": str(e), "response": "Forecast failed."}