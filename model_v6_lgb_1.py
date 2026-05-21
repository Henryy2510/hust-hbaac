
# %% [0] Config
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError("lightgbm is required") from e

VERSION = "v6_lgb_1"
TRAIN_PATH = "old/train_fixed_ffill.csv"
SUB_PATH = "sample_submission.csv"
OUT_PATH = "submission_v6_lgb_1.csv"
METRICS_PATH = "metrics_v6_lgb_1.json"
PROFILE = {"fill": "ffill", "outlier": False}

SEED = 42
VAL_DAYS = 28
HORIZON = 56
HISTORY_KEEP = 430
EARLY_STOP_DAYS = 60
EARLY_STOPPING_ROUNDS = 80
USE_EARLY_STOPPING = True

np.random.seed(SEED)
print("version", VERSION, "profile", PROFILE)

# %% [1] Load and daily aggregate
FLOAT_COLS = ["Quantity", "UnitPrice", "SalesAmount", "Unit Cost", "Cost Amount"]
USECOLS = ["Date", "ItemCode"] + FLOAT_COLS

df_raw = pd.read_csv(TRAIN_PATH, usecols=USECOLS, dtype={"ItemCode": "category"}, parse_dates=["Date"])
for col in FLOAT_COLS:
    df_raw[col] = df_raw[col].astype(str).str.replace(",", ".", regex=False).astype("float32")

# Prices from positive-sale rows only. Return rows still affect net quantity/revenue/cost sums.
df_raw["UnitPrice_clean"] = np.where(df_raw["Quantity"] > 0, np.abs(df_raw["UnitPrice"]), np.nan).astype("float32")
df_raw["UnitCost_clean"] = np.where(df_raw["Quantity"] > 0, np.abs(df_raw["Unit Cost"]), np.nan).astype("float32")
df_raw["sales_err"] = (df_raw["SalesAmount"].astype("float64") - df_raw["Quantity"].astype("float64") * df_raw["UnitPrice"].astype("float64")).abs().astype("float32")
df_raw["cost_err"] = (df_raw["Cost Amount"].astype("float64") - df_raw["Quantity"].astype("float64") * df_raw["Unit Cost"].astype("float64")).abs().astype("float32")

daily = (
    df_raw.groupby(["Date", "ItemCode"], observed=True, as_index=False)
    .agg(
        Quantity=("Quantity", "sum"),
        SalesAmount=("SalesAmount", "sum"),
        CostAmount=("Cost Amount", "sum"),
        UnitPrice=("UnitPrice_clean", "mean"),
        UnitCost=("UnitCost_clean", "mean"),
        sales_err=("sales_err", "mean"),
        cost_err=("cost_err", "mean"),
    )
    .sort_values(["ItemCode", "Date"])
)
daily["y"] = daily["Quantity"].clip(lower=0).astype("float32")
daily["profit"] = (daily["SalesAmount"] - daily["CostAmount"]).astype("float32")

business_dates = pd.Index(sorted(daily["Date"].unique()))
all_calendar_dates = pd.date_range(business_dates.min(), business_dates.max(), freq="D")
all_skus = daily["ItemCode"].cat.categories if str(daily["ItemCode"].dtype) == "category" else pd.Index(daily["ItemCode"].unique())

panel = pd.MultiIndex.from_product([business_dates, all_skus], names=["Date", "ItemCode"]).to_frame(index=False)
panel["ItemCode"] = panel["ItemCode"].astype("category")
panel = panel.merge(daily, on=["Date", "ItemCode"], how="left").sort_values(["ItemCode", "Date"]).reset_index(drop=True)
for c in ["Quantity", "SalesAmount", "CostAmount", "y", "profit", "sales_err", "cost_err"]:
    panel[c] = panel[c].fillna(0).astype("float32")

assert panel["Date"].nunique() == len(business_dates)
assert panel["Date"].nunique() < len(all_calendar_dates)
print("raw", df_raw.shape, "daily", daily.shape, "panel", panel.shape)
print("business_dates", len(business_dates), "calendar_dates", len(all_calendar_dates), "sundays_present", int((business_dates.dayofweek == 6).sum()))

# %% [2] Value filling by dataset profile
def fill_value_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if PROFILE["fill"] == "ffill":
        for c in ["UnitPrice", "UnitCost"]:
            out[c] = out[c].replace(0, np.nan)
            out[c] = out.groupby("ItemCode", observed=True, sort=False)[c].ffill()
            out[c] = out[c].fillna(out[c].median()).astype("float32")
    else:
        for c in ["UnitPrice", "UnitCost"]:
            med = out.groupby("ItemCode", observed=True)[c].transform(lambda s: s.replace(0, np.nan).median())
            out[c] = out[c].replace(0, np.nan).fillna(med).fillna(out[c].replace(0, np.nan).median()).astype("float32")

    if PROFILE["outlier"]:
        for c in ["UnitPrice", "UnitCost"]:
            lo = out[c].quantile(0.001)
            hi = out[c].quantile(0.999)
            out[f"{c}_capped"] = out[c].clip(lo, hi).astype("float32")
    else:
        out["UnitPrice_capped"] = out["UnitPrice"].astype("float32")
        out["UnitCost_capped"] = out["UnitCost"].astype("float32")

    out["margin"] = (out["UnitPrice"] - out["UnitCost"]).astype("float32")
    out["margin_pct"] = (out["margin"] / out["UnitPrice"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    out["margin_capped"] = (out["UnitPrice_capped"] - out["UnitCost_capped"]).astype("float32")
    return out

panel = fill_value_columns(panel)
assert float(panel["UnitPrice"].min()) >= 0
assert float(panel["UnitCost"].min()) >= 0

# %% [3] Metrics and leakage-safe static stats
def active_scale(s: pd.Series) -> float:
    active = s[s > 0]
    base = active if len(active) > 1 else s
    d = base.diff().dropna()
    val = float((d ** 2).mean()) if len(d) else 1.0
    return val if np.isfinite(val) and val > 0 else 1.0


def build_metric_artifacts(train_panel: pd.DataFrame):
    scale = train_panel.groupby("ItemCode", observed=True)["y"].apply(active_scale).astype("float64")
    profit = train_panel.groupby("ItemCode", observed=True)["profit"].sum().clip(lower=0).astype("float64")
    weights = profit / profit.sum() if float(profit.sum()) > 0 else pd.Series(1.0 / len(profit), index=profit.index)
    zero_weight_skus = set(weights[weights <= 0].index.astype(str))
    row_weight_map = (weights / scale.reindex(weights.index).fillna(1.0)).replace([np.inf, -np.inf], 0).fillna(0)
    return scale, weights, row_weight_map, zero_weight_skus


def wrmsse_score(eval_df: pd.DataFrame, scale: pd.Series, weights: pd.Series) -> float:
    mse = eval_df.groupby("ItemCode", observed=True).apply(lambda x: np.mean((x["y"].values - x["pred"].values) ** 2), include_groups=False)
    common = mse.index.intersection(scale.index).intersection(weights.index)
    if len(common) == 0:
        return float("nan")
    rmsse = np.sqrt((mse.loc[common] / scale.loc[common]).clip(lower=0))
    return float((rmsse * weights.loc[common]).sum())


def build_sku_stats(base_panel: pd.DataFrame) -> pd.DataFrame:
    g = base_panel.groupby("ItemCode", observed=True)["y"]
    stats = g.agg(["mean", "std", "sum"]).rename(columns={"mean": "sku_y_mean", "std": "sku_y_std", "sum": "sku_y_sum"})
    stats["sku_active_ratio"] = g.apply(lambda s: float(np.mean(s > 0)))
    stats["sku_age_days"] = g.size().astype("float32")
    for w in [28, 56, 84]:
        stats[f"sku_recent_mean_{w}"] = base_panel.groupby("ItemCode", observed=True).tail(w).groupby("ItemCode", observed=True)["y"].mean()
    profit = base_panel.groupby("ItemCode", observed=True)["profit"].sum().clip(lower=0)
    weight = profit / profit.sum() if float(profit.sum()) > 0 else pd.Series(1.0 / len(profit), index=profit.index)
    stats["profit_weight"] = weight
    stats["profit_weight_bucket"] = pd.qcut(stats["profit_weight"].rank(method="first"), 10, labels=False).astype("int8")
    stats = stats.fillna(0).reset_index()
    return stats


def build_croston_stats(base_panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sku, s in base_panel.groupby("ItemCode", observed=True)["y"]:
        vals = s.values.astype("float64")
        nz = np.flatnonzero(vals > 0)
        if len(nz) == 0:
            size, interval, rate = 0.0, 9999.0, 0.0
        elif len(nz) == 1:
            size, interval, rate = float(vals[nz].mean()), 9999.0, 0.0
        else:
            size = float(vals[nz].mean())
            interval = float(np.diff(nz).mean())
            rate = size / interval if interval > 0 else 0.0
        rows.append((sku, size, interval, rate))
    return pd.DataFrame(rows, columns=["ItemCode", "croston_size", "croston_interval", "croston_rate"])

# %% [4] Feature engineering
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dow"] = out["Date"].dt.dayofweek.astype("int8")
    out["dom"] = out["Date"].dt.day.astype("int8")
    out["month"] = out["Date"].dt.month.astype("int8")
    out["quarter"] = out["Date"].dt.quarter.astype("int8")
    out["woy"] = out["Date"].dt.isocalendar().week.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int8")
    out["is_month_start"] = out["Date"].dt.is_month_start.astype("int8")
    out["is_month_end"] = out["Date"].dt.is_month_end.astype("int8")
    return out


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["ItemCode", "Date"]).reset_index(drop=True)
    g = out.groupby("ItemCode", observed=True, sort=False)["y"]
    for lag in [1, 2, 3, 7, 14, 21, 28, 56, 84, 91, 182, 364]:
        out[f"lag_{lag}"] = g.shift(lag).astype("float32")
    shifted = g.shift(1)
    for w in [7, 14, 28, 56, 84, 365]:
        roll = shifted.groupby(out["ItemCode"], observed=True).rolling(w)
        out[f"rmean_{w}"] = roll.mean().reset_index(level=0, drop=True).astype("float32")
        out[f"rstd_{w}"] = roll.std().reset_index(level=0, drop=True).astype("float32")
        out[f"rsum_{w}"] = roll.sum().reset_index(level=0, drop=True).astype("float32")
        out[f"rmax_{w}"] = roll.max().reset_index(level=0, drop=True).astype("float32")
        out[f"rnz_{w}"] = roll.apply(lambda x: np.count_nonzero(x > 0), raw=True).reset_index(level=0, drop=True).astype("float32")
        out[f"zero_ratio_{w}"] = (1.0 - out[f"rnz_{w}"] / float(w)).astype("float32")
        out[f"sale_prob_{w}"] = (out[f"rnz_{w}"] / float(w)).astype("float32")
        out[f"avg_nonzero_{w}"] = (out[f"rsum_{w}"] / out[f"rnz_{w}"].replace(0, np.nan)).fillna(0).astype("float32")
    sale = out["y"] > 0
    idx = out.groupby("ItemCode", observed=True, sort=False).cumcount()
    last_idx = idx.where(sale).groupby(out["ItemCode"], observed=True).ffill().groupby(out["ItemCode"], observed=True).shift(1)
    out["days_since_last_sale"] = (idx - last_idx).fillna(9999).clip(0, 9999).astype("int16")
    last_nonzero = out["y"].where(sale).groupby(out["ItemCode"], observed=True).ffill()
    out["last_nonzero_qty"] = last_nonzero.groupby(out["ItemCode"], observed=True).shift(1).fillna(0).astype("float32")
    return out


def add_value_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby("ItemCode", observed=True, sort=False)
    if PROFILE["fill"] == "ffill":
        for c in ["UnitPrice", "UnitCost", "margin", "margin_pct", "UnitPrice_capped", "UnitCost_capped", "margin_capped"]:
            for lag in [1, 7, 28]:
                out[f"{c}_lag_{lag}"] = g[c].shift(lag).astype("float32")
            out[f"{c}_chg_7"] = (out[c] - g[c].shift(7)).astype("float32")
            shifted_c = g[c].shift(1)
            out[f"{c}_rmean_28"] = shifted_c.groupby(out["ItemCode"], observed=True).rolling(28).mean().reset_index(level=0, drop=True).astype("float32")
    else:
        for c in ["UnitPrice", "UnitCost", "margin", "margin_pct"]:
            med = out.groupby("ItemCode", observed=True)[c].transform("median").astype("float32")
            out[f"sku_{c}_median"] = med
            out[f"{c}_vs_sku_median"] = (out[c] - med).astype("float32")
    if PROFILE["outlier"]:
        out["sales_err_ratio"] = (out["sales_err"] / out["SalesAmount"].abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10).astype("float32")
        out["cost_err_ratio"] = (out["cost_err"] / out["CostAmount"].abs().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10).astype("float32")
    return out


def make_features(df: pd.DataFrame, sku_stats: pd.DataFrame, croston_stats: pd.DataFrame) -> pd.DataFrame:
    out = add_calendar_features(df)
    out = add_demand_features(out)
    out = add_value_features(out)
    out = out.merge(sku_stats, on="ItemCode", how="left").merge(croston_stats, on="ItemCode", how="left")
    return out.fillna(0)

# %% [5] Model wrappers and validation frames
class ModelWrapper:
    def __init__(self, name, model, log_target=False):
        self.name = name
        self.model = model
        self.log_target = log_target

    def fit(self, X, y, sample_weight=None, eval_set=None, eval_sample_weight=None):
        yy = np.log1p(y) if self.log_target else y
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if eval_set is not None and isinstance(self.model, lgb.LGBMRegressor):
            X_eval, y_eval = eval_set
            yy_eval = np.log1p(y_eval) if self.log_target else y_eval
            fit_kwargs["eval_set"] = [(X_eval, yy_eval)]
            if eval_sample_weight is not None:
                fit_kwargs["eval_sample_weight"] = [eval_sample_weight]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                lgb.log_evaluation(500),
            ]
        try:
            self.model.fit(X, yy, **fit_kwargs)
        except TypeError:
            self.model.fit(X, yy)
        return self

    def predict(self, X):
        p = self.model.predict(X)
        if self.log_target:
            p = np.expm1(p)
        return np.clip(p, 0, None)


def candidate_models():
    models = [
        ModelWrapper("lgb_tweedie_13", lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.3, n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 1, n_jobs=-1, verbosity=-1)),
        ModelWrapper("lgb_tweedie_15", lgb.LGBMRegressor(objective="tweedie", tweedie_variance_power=1.5, n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 2, n_jobs=-1, verbosity=-1)),
        ModelWrapper("lgb_poisson", lgb.LGBMRegressor(objective="poisson", n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 4, n_jobs=-1, verbosity=-1)),
        ModelWrapper("lgb_huber_log1p", lgb.LGBMRegressor(objective="huber", alpha=0.9, n_estimators=2500, learning_rate=0.03, num_leaves=127, min_child_samples=60, subsample=0.85, colsample_bytree=0.85, reg_alpha=0.03, reg_lambda=0.8, random_state=SEED + 5, n_jobs=-1, verbosity=-1), log_target=True),
        ModelWrapper("lgb_rf_tweedie", lgb.LGBMRegressor(boosting_type="rf", objective="tweedie", tweedie_variance_power=1.3, n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=40, bagging_fraction=0.75, bagging_freq=1, feature_fraction=0.75, random_state=SEED + 6, n_jobs=-1, verbosity=-1)),
    ]
    return models

valid_business_dates = business_dates[-VAL_DAYS:]
valid_start = valid_business_dates[0]
calendar_valid_dates = pd.date_range(valid_business_dates[0], valid_business_dates[-1], freq="D")
train_panel = panel[panel["Date"] < valid_start].copy()
valid_panel = panel[panel["Date"].isin(valid_business_dates)].copy()
assert valid_panel["Date"].nunique() == 28
print("valid business", valid_business_dates[0].date(), "->", valid_business_dates[-1].date())
print("calendar valid present", int(pd.Index(calendar_valid_dates).isin(business_dates).sum()), "of", len(calendar_valid_dates))

scale, weights, row_weight_map, zero_weight_skus = build_metric_artifacts(train_panel)
sku_stats_train = build_sku_stats(train_panel)
croston_train = build_croston_stats(train_panel)
train_feat = make_features(train_panel, sku_stats_train, croston_train)
valid_context = pd.concat([
    train_panel.groupby("ItemCode", observed=True).tail(HISTORY_KEEP),
    valid_panel,
], ignore_index=True)
valid_feat_all = make_features(valid_context, sku_stats_train, croston_train)
valid_feat = valid_feat_all[valid_feat_all["Date"].isin(valid_business_dates)].copy()

exclude = {"Date", "ItemCode", "Quantity", "SalesAmount", "CostAmount", "UnitPrice", "UnitCost", "profit", "y"}
feature_cols = [c for c in train_feat.columns if c not in exclude]
X_train = train_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_train = train_feat["y"].values.astype("float32")
X_valid = valid_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_valid = valid_feat["y"].values.astype("float32")
sample_weight = train_feat["ItemCode"].map(row_weight_map).fillna(0).values.astype("float32")

active_train_mask = sample_weight > 0
es_start_idx = max(0, len(business_dates[business_dates < valid_start]) - EARLY_STOP_DAYS)
early_stop_start = business_dates[business_dates < valid_start][es_start_idx]
fit_mask = active_train_mask & (train_feat["Date"].values < early_stop_start)
es_mask = active_train_mask & (train_feat["Date"].values >= early_stop_start)
print("train rows", int(active_train_mask.sum()), "fit rows", int(fit_mask.sum()), "early_stop rows", int(es_mask.sum()), "early_stop_start", pd.Timestamp(early_stop_start).date())

X_fit = X_train.loc[fit_mask]
y_fit = y_train[fit_mask]
sw_fit = sample_weight[fit_mask]
X_es = X_train.loc[es_mask]
y_es = y_train[es_mask]
sw_es = sample_weight[es_mask]

# %% [6] Train candidates and score business/calendar validation
state_cols = ["Date", "ItemCode", "Quantity", "SalesAmount", "CostAmount", "UnitPrice", "UnitCost", "sales_err", "cost_err", "UnitPrice_capped", "UnitCost_capped", "margin", "margin_pct", "margin_capped", "profit", "y"]


def calendar_eval_from_business(pred_values):
    pred_df = valid_feat[["Date", "ItemCode", "y"]].copy()
    pred_df["pred"] = pred_values
    idx = pd.MultiIndex.from_product([calendar_valid_dates, all_skus], names=["Date", "ItemCode"]).to_frame(index=False)
    idx["ItemCode"] = idx["ItemCode"].astype("category")
    out = idx.merge(pred_df, on=["Date", "ItemCode"], how="left")
    out["y"] = out["y"].fillna(0).astype("float32")
    out["pred"] = out["pred"].fillna(0).astype("float32")
    return out


def calendar_eval_from_pred_df(pred_df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product([calendar_valid_dates, all_skus], names=["Date", "ItemCode"]).to_frame(index=False)
    idx["ItemCode"] = idx["ItemCode"].astype("category")
    actual = valid_panel[["Date", "ItemCode", "y"]]
    out = idx.merge(actual, on=["Date", "ItemCode"], how="left").merge(pred_df[["Date", "ItemCode", "pred"]], on=["Date", "ItemCode"], how="left")
    out["y"] = out["y"].fillna(0).astype("float32")
    out["pred"] = out["pred"].fillna(0).astype("float32")
    return out


def recursive_forecast(model_dict: dict, blend_names: list, blend_weights: np.ndarray, dates, base_panel: pd.DataFrame, label: str) -> pd.DataFrame:
    state = base_panel[state_cols].copy().sort_values(["ItemCode", "Date"]).reset_index(drop=True)
    pred_rows = []
    for d in dates:
        recent = state.groupby("ItemCode", observed=True).tail(HISTORY_KEEP).copy()
        last_vals = state.groupby("ItemCode", observed=True).tail(1).set_index("ItemCode")
        step = pd.DataFrame({"Date": d, "ItemCode": all_skus})
        step["ItemCode"] = step["ItemCode"].astype("category")
        step = step.join(last_vals[["UnitPrice", "UnitCost", "UnitPrice_capped", "UnitCost_capped", "margin", "margin_pct", "margin_capped"]], on="ItemCode")
        for c in ["Quantity", "SalesAmount", "CostAmount", "sales_err", "cost_err", "profit"]:
            step[c] = 0.0
        step["y"] = np.nan
        step = fill_value_columns(step)

        feat_state = make_features(pd.concat([recent, step[state_cols]], ignore_index=True), sku_stats_train, croston_train)
        cur = feat_state[feat_state["Date"] == d].copy()
        X_cur = cur[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
        mat = np.column_stack([model_dict[n].predict(X_cur) for n in blend_names])
        pred = np.clip(mat @ blend_weights, 0, None)
        sku_str = cur["ItemCode"].astype(str).values
        pred[np.isin(sku_str, list(zero_weight_skus))] = 0.0

        step["y"] = pred.astype("float32")
        state = pd.concat([state, step[state_cols]], ignore_index=True).sort_values(["ItemCode", "Date"])
        state = state.groupby("ItemCode", observed=True).tail(HISTORY_KEEP + 10).reset_index(drop=True)
        pred_rows.append(pd.DataFrame({"Date": d, "ItemCode": cur["ItemCode"].values, "pred": pred}))
        print(label, pd.Timestamp(d).date(), "q50", float(np.quantile(pred, .5)), "q99", float(np.quantile(pred, .99)))
    out = pd.concat(pred_rows, ignore_index=True)
    return out.merge(valid_panel[["Date", "ItemCode", "y"]], on=["Date", "ItemCode"], how="left")


trained = {}
valid_preds = {}
scores = []
for model in candidate_models():
    print("training", model.name)
    eval_set = (X_es, y_es) if USE_EARLY_STOPPING and len(X_es) else None
    eval_sw = sw_es if USE_EARLY_STOPPING and len(X_es) else None
    model.fit(X_fit, y_fit, sample_weight=sw_fit, eval_set=eval_set, eval_sample_weight=eval_sw)
    trained[model.name] = model
    pred = model.predict(X_valid)
    valid_preds[model.name] = pred
    business_df = valid_feat[["Date", "ItemCode", "y"]].copy()
    business_df["pred"] = pred
    cal_df = calendar_eval_from_business(pred)
    scores.append({
        "model": model.name,
        "business_rmse": float(np.sqrt(mean_squared_error(y_valid, pred))),
        "business_wrmsse": wrmsse_score(business_df, scale, weights),
        "calendar_wrmsse": wrmsse_score(cal_df, scale, weights),
        "best_iteration": int(getattr(model.model, "best_iteration_", 0) or 0),
    })
    print(scores[-1])

score_df = pd.DataFrame(scores).sort_values(["calendar_wrmsse", "business_wrmsse"])
print(score_df)

# %% [7] Blend optimization and one-pass recursive validation
def optimize_blend(preds):
    names = list(preds)
    P = np.column_stack([preds[n] for n in names])

    def obj(w):
        pred = np.clip(P @ w, 0, None)
        pred_df = valid_panel[["Date", "ItemCode", "y"]].copy()
        pred_df["pred"] = pred
        return wrmsse_score(calendar_eval_from_pred_df(pred_df), scale, weights)

    x0 = np.ones(len(names)) / len(names)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, 1)] * len(names), constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},), options={"maxiter": 200})
    w = res.x if res.success else x0
    w = np.clip(w, 0, None)
    w = w / w.sum() if w.sum() > 0 else x0
    return names, w, obj(w)

names, global_w, leaked_global_calendar_wr = optimize_blend(valid_preds)
recursive_blend_df = recursive_forecast(trained, names, global_w, valid_business_dates, train_panel, "recursive_valid blend")
recursive_blend_cal = calendar_eval_from_pred_df(recursive_blend_df)
global_recursive_business_rmse = float(np.sqrt(mean_squared_error(recursive_blend_df["y"].values, recursive_blend_df["pred"].values)))
global_recursive_business_wr = wrmsse_score(recursive_blend_df, scale, weights)
global_recursive_wr = wrmsse_score(recursive_blend_cal, scale, weights)
print(
    "global blend",
    dict(zip(names, map(float, global_w))),
    "leaked_calendar",
    leaked_global_calendar_wr,
    "recursive_business",
    global_recursive_business_wr,
    "recursive_calendar",
    global_recursive_wr,
)
best_iterations = {
    name: int(getattr(model.model, "best_iteration_", 0) or model.model.get_params().get("n_estimators", 0))
    for name, model in trained.items()
}
print("best iterations", best_iterations)

# %% [8] Refit full history
scale_full, weights_full, row_weight_map_full, zero_weight_skus_full = build_metric_artifacts(panel)
sku_stats_full = build_sku_stats(panel)
croston_full = build_croston_stats(panel)
full_feat = make_features(panel, sku_stats_full, croston_full)
X_full = full_feat[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
y_full = full_feat["y"].values.astype("float32")
sample_weight_full = full_feat["ItemCode"].map(row_weight_map_full).fillna(0).values.astype("float32")
active_full_mask = sample_weight_full > 0
print("full train rows", int(active_full_mask.sum()))

refit = {}
for model in candidate_models():
    print("refit", model.name)
    best_iter = best_iterations.get(model.name, 0)
    if best_iter > 0 and isinstance(model.model, lgb.LGBMRegressor):
        model.model.set_params(n_estimators=best_iter)
    model.fit(X_full.loc[active_full_mask], y_full[active_full_mask], sample_weight=sample_weight_full[active_full_mask])
    refit[model.name] = model

# %% [9] Forecast 56 calendar days
state = panel[state_cols].copy().sort_values(["ItemCode", "Date"]).reset_index(drop=True)
future_dates = pd.date_range(business_dates.max() + pd.Timedelta(days=1), periods=HORIZON, freq="D")
pred_rows = []
closed_days = []
for d in future_dates:
    is_closed = d.dayofweek == 6
    if is_closed:
        closed_days.append(d)
        pred = np.zeros(len(all_skus), dtype="float32")
        pred_rows.append(pd.DataFrame({"Date": d, "ItemCode": all_skus, "pred": pred}))
        print("forecast closed", d.date())
        continue

    recent = state.groupby("ItemCode", observed=True).tail(HISTORY_KEEP).copy()
    last_vals = state.groupby("ItemCode", observed=True).tail(1).set_index("ItemCode")
    step = pd.DataFrame({"Date": d, "ItemCode": all_skus})
    step["ItemCode"] = step["ItemCode"].astype("category")
    step = step.join(last_vals[["UnitPrice", "UnitCost", "UnitPrice_capped", "UnitCost_capped", "margin", "margin_pct", "margin_capped"]], on="ItemCode")
    for c in ["Quantity", "SalesAmount", "CostAmount", "sales_err", "cost_err", "profit"]:
        step[c] = 0.0
    step["y"] = np.nan
    step = fill_value_columns(step)

    feat_state = make_features(pd.concat([recent, step[state_cols]], ignore_index=True), sku_stats_full, croston_full)
    cur = feat_state[feat_state["Date"] == d].copy()
    X_cur = cur[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    mat = np.column_stack([refit[n].predict(X_cur) for n in names])
    pred = np.clip(mat @ global_w, 0, None)

    sku_str = cur["ItemCode"].astype(str).values
    zero_mask = np.isin(sku_str, list(zero_weight_skus_full))
    pred[zero_mask] = 0.0

    step["y"] = pred.astype("float32")
    state = pd.concat([state, step[state_cols]], ignore_index=True).sort_values(["ItemCode", "Date"])
    state = state.groupby("ItemCode", observed=True).tail(HISTORY_KEEP + 10).reset_index(drop=True)
    pred_rows.append(pd.DataFrame({"Date": d, "ItemCode": cur["ItemCode"].values, "pred": pred}))
    print("forecast open", d.date(), "q50", float(np.quantile(pred, .5)), "q99", float(np.quantile(pred, .99)), "near0", float(np.mean(pred < 1e-6)))

pred56 = pd.concat(pred_rows, ignore_index=True)
print("future open", HORIZON - len(closed_days), "closed", len(closed_days))
print("pred quantiles", pred56["pred"].quantile([0, .5, .9, .99]).to_dict())

# %% [10] Submission and metrics
sample = pd.read_csv(SUB_PATH)
fcols = [f"F{i}" for i in range(1, 29)]
sku_to_vals = pred56.sort_values(["ItemCode", "Date"]).groupby("ItemCode", observed=True)["pred"].apply(list).to_dict()
rows = []
missing_ids = []
for rid in sample["id"]:
    if rid.endswith("_validation"):
        sku = rid[: -len("_validation")]
        vals = sku_to_vals.get(sku)
        block = vals[:28] if vals is not None else None
    elif rid.endswith("_evaluation"):
        sku = rid[: -len("_evaluation")]
        vals = sku_to_vals.get(sku)
        block = vals[28:56] if vals is not None else None
    else:
        raise ValueError(f"Bad submission id suffix: {rid}")
    if block is None:
        missing_ids.append(rid)
        block = [0.0] * 28
    rows.append([rid] + [float(max(0.0, v)) for v in block])

sub = pd.DataFrame(rows, columns=["id"] + fcols)
assert not missing_ids, f"Missing predictions for ids: {missing_ids[:5]}"
assert sub.shape == sample.shape
assert sub["id"].is_unique
assert set(sub["id"]) == set(sample["id"])
assert np.isfinite(sub[fcols].to_numpy()).all()
assert (sub[fcols].to_numpy() >= 0).all()
sub.to_csv(OUT_PATH, index=False)

metrics = {
    "version": VERSION,
    "train_path": TRAIN_PATH,
    "profile": PROFILE,
    "business_dates": int(len(business_dates)),
    "calendar_dates": int(len(all_calendar_dates)),
    "business_valid_dates": int(len(valid_business_dates)),
    "calendar_valid_dates": int(len(calendar_valid_dates)),
    "calendar_valid_present_dates": int(pd.Index(calendar_valid_dates).isin(business_dates).sum()),
    "closed_future_days": int(len(closed_days)),
    "zero_weight_sku_count": int(len(zero_weight_skus_full)),
    "leaked_validation_scores": scores,
    "global_weights": dict(zip(names, map(float, global_w))),
    "leaked_global_calendar_wrmsse": float(leaked_global_calendar_wr),
    "global_recursive_business_rmse": float(global_recursive_business_rmse),
    "global_recursive_business_wrmsse": float(global_recursive_business_wr),
    "global_recursive_calendar_wrmsse": float(global_recursive_wr),
    "best_iterations": best_iterations,
    "pred_quantiles": {str(k): float(v) for k, v in pred56["pred"].quantile([0, .5, .9, .99]).to_dict().items()},
    "pred_near_zero_share": float((pred56["pred"] < 1e-6).mean()),
    "feature_count": int(len(feature_cols)),
    "model_option": "v6_lgb_efficient_recursive_valid",
    "history_keep": int(HISTORY_KEEP),
    "early_stop_days": int(EARLY_STOP_DAYS),
    "early_stopping_rounds": int(EARLY_STOPPING_ROUNDS),
    "feature_cols": feature_cols,
}
Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print("saved", OUT_PATH, METRICS_PATH)
