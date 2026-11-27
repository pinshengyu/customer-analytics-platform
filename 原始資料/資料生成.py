import pandas as pd
import numpy as np
import random
from pathlib import Path

# ============================================================
# RAW DATA GENERATOR V6
# 原始資料範例校準版
#
# 參考資料：
# 1. online_retail_II.csv
# 2. Online Retail.xlsx
# 3. multi_touch_attribution_data.csv
#
# 原則：
# - 真實範例資料只用來取得「分布與相對行為特徵」
# - 不直接複製原始顧客、訂單或商品
# - 專題自訂欄位與 2023~2025 時間架構仍由本生成器控制
# ============================================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ============================================================
# Parameters
# ============================================================

N_CUSTOMERS = 20_000
N_PRODUCTS = 1_000
N_CAMPAIGNS = 50

START_TS = pd.Timestamp("2023-01-01")
END_TS = pd.Timestamp("2025-12-31")

OBSERVATION_END = pd.Timestamp("2024-12-31")
FUTURE_START = pd.Timestamp("2025-01-01")
FUTURE_END = pd.Timestamp("2025-12-31")

REFERRAL_RATE = 0.30
TARGET_CHURN_RATE = 0.20

# 參考 Online Retail 一張 Invoice 的商品明細很多，
# 若完全照搬會使 20,000 客戶資料量過度膨脹。
# 因此保留「明細數分布形狀」，但縮放到本專題合理範圍。
TARGET_AVG_LINES_PER_ORDER = 3.20
MAX_LINES_PER_ORDER = 8
MAX_QUANTITY_PER_LINE = 8

# ============================================================
# Project Root / Paths
# ============================================================

def find_project_root():
    """
    從資料生成.py 所在位置一路往上找「原始資料範例」資料夾。
    不依賴固定 parent 層數。
    """
    current = Path(__file__).resolve().parent

    for parent in [current, *current.parents]:
        if (parent / "原始資料範例").is_dir():
            return parent

    raise FileNotFoundError(
        "找不到專題根目錄。\n"
        "請確認資料生成.py 位於專題資料夾內，"
        "且專題根目錄存在「原始資料範例」資料夾。"
    )


PROJECT_DIR = find_project_root()
REFERENCE_DIR = PROJECT_DIR / "原始資料範例"
OUTPUT_DIR = PROJECT_DIR / "RAW_Data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_reference_file(stem_candidates, suffix_candidates):
    """
    在「原始資料範例」中以檔名（不分大小寫）自動尋找檔案。
    """
    all_files = [
        p for p in REFERENCE_DIR.iterdir()
        if p.is_file()
    ]

    normalized = {
        (p.stem.strip().lower(), p.suffix.strip().lower()): p
        for p in all_files
    }

    for stem in stem_candidates:
        for suffix in suffix_candidates:
            key = (stem.strip().lower(), suffix.strip().lower())
            if key in normalized:
                return normalized[key]

    raise FileNotFoundError(
        "找不到原始資料範例。\n"
        f"預期檔名之一：{stem_candidates}\n"
        f"允許格式：{suffix_candidates}\n"
        f"目前資料夾：{REFERENCE_DIR}"
    )


ONLINE_RETAIL_II_PATH = find_reference_file(
    ["online_retail_II"],
    [".csv"]
)

MULTI_TOUCH_PATH = find_reference_file(
    ["multi_touch_attribution_data"],
    [".csv"]
)

ONLINE_RETAIL_PATH = find_reference_file(
    ["Online Retail", "online retail"],
    [".xlsx", ".xls"]
)

print("=" * 78)
print("RAW DATA GENERATOR V6 - REFERENCE CALIBRATED")
print("=" * 78)
print("Project Root :", PROJECT_DIR)
print("Reference Dir:", REFERENCE_DIR)
print("Output Dir   :", OUTPUT_DIR)
print()
print("Reference Files:")
print(" -", ONLINE_RETAIL_II_PATH.name)
print(" -", MULTI_TOUCH_PATH.name)
print(" -", ONLINE_RETAIL_PATH.name)

# ============================================================
# Helpers
# ============================================================

def safe_probability(value):
    return float(np.clip(value, 0.0, 1.0))


def random_date(start, end):
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize()

    if start >= end:
        return start

    days = (end - start).days

    return start + pd.Timedelta(
        days=np.random.randint(0, days + 1)
    )


def weighted_choice(values, probabilities):
    values = np.asarray(values)
    probabilities = np.asarray(probabilities, dtype=float)

    probabilities = probabilities / probabilities.sum()

    return np.random.choice(
        values,
        p=probabilities
    )


# ============================================================
# 0. Load and Calibrate Reference Data
# ============================================================

print("\n[0/8] Loading reference datasets...")

retail_ii = pd.read_csv(
    ONLINE_RETAIL_II_PATH,
    low_memory=False
)

online_retail = pd.read_excel(
    ONLINE_RETAIL_PATH
)

mta = pd.read_csv(
    MULTI_TOUCH_PATH,
    low_memory=False
)


def clean_retail_reference(
    df,
    invoice_col,
    stock_col,
    quantity_col,
    date_col,
    price_col,
    customer_col
):
    """
    將兩份 Online Retail 統一成相同欄位後清理：
    - 排除取消單
    - 排除 CustomerID 缺失
    - Quantity <= 0 排除
    - Price <= 0 排除
    - 日期錯誤排除
    """
    x = df[
        [
            invoice_col,
            stock_col,
            quantity_col,
            date_col,
            price_col,
            customer_col
        ]
    ].copy()

    x.columns = [
        "Invoice",
        "StockCode",
        "Quantity",
        "InvoiceDate",
        "Price",
        "CustomerID"
    ]

    x["Invoice"] = x["Invoice"].astype(str)
    x["StockCode"] = x["StockCode"].astype(str)

    x["InvoiceDate"] = pd.to_datetime(
        x["InvoiceDate"],
        errors="coerce"
    )

    x["Quantity"] = pd.to_numeric(
        x["Quantity"],
        errors="coerce"
    )

    x["Price"] = pd.to_numeric(
        x["Price"],
        errors="coerce"
    )

    x["CustomerID"] = pd.to_numeric(
        x["CustomerID"],
        errors="coerce"
    )

    x = x[
        x["InvoiceDate"].notna()
        & x["CustomerID"].notna()
        & x["Quantity"].gt(0)
        & x["Price"].gt(0)
    ].copy()

    # Online Retail 取消單通常以 C 開頭
    x = x[
        ~x["Invoice"].str.upper().str.startswith("C")
    ].copy()

    return x


reference_retail_1 = clean_retail_reference(
    online_retail,
    invoice_col="InvoiceNo",
    stock_col="StockCode",
    quantity_col="Quantity",
    date_col="InvoiceDate",
    price_col="UnitPrice",
    customer_col="CustomerID"
)

reference_retail_2 = clean_retail_reference(
    retail_ii,
    invoice_col="Invoice",
    stock_col="StockCode",
    quantity_col="Quantity",
    date_col="InvoiceDate",
    price_col="Price",
    customer_col="Customer ID"
)

reference_retail = pd.concat(
    [
        reference_retail_1,
        reference_retail_2
    ],
    ignore_index=True
)

reference_retail["Revenue"] = (
    reference_retail["Quantity"]
    * reference_retail["Price"]
)

if reference_retail.empty:
    raise ValueError(
        "兩份 Online Retail 範例清理後沒有可用交易資料。"
    )

if mta.empty:
    raise ValueError(
        "multi_touch_attribution_data.csv 沒有資料。"
    )

print(
    "    Valid Retail Rows:",
    f"{len(reference_retail):,}"
)

print(
    "    Valid Retail Transactions:",
    f"{reference_retail['Invoice'].nunique():,}"
)

print(
    "    Valid Retail Customers:",
    f"{reference_retail['CustomerID'].nunique():,}"
)

# ------------------------------------------------------------
# Reference: lines per transaction
# ------------------------------------------------------------

ref_lines_per_order = (
    reference_retail
    .groupby("Invoice")
    .size()
    .astype(int)
)

ref_lines_mean = float(
    ref_lines_per_order.mean()
)

ref_lines_median = float(
    ref_lines_per_order.median()
)

# ------------------------------------------------------------
# Reference: Quantity distribution
# 避免原始資料極端批發值控制生成結果
# ------------------------------------------------------------

ref_quantity = reference_retail[
    "Quantity"
].to_numpy(dtype=float)

ref_quantity_upper = float(
    np.quantile(
        ref_quantity,
        0.99
    )
)

ref_quantity_clean = np.clip(
    ref_quantity,
    1,
    ref_quantity_upper
)

ref_quantity_median = max(
    1.0,
    float(
        np.median(
            ref_quantity_clean
        )
    )
)

# ------------------------------------------------------------
# Reference: Price distribution
# 只採「相對位置」，不直接使用英鎊價格
# ------------------------------------------------------------

ref_price = reference_retail[
    "Price"
].to_numpy(dtype=float)

ref_price_upper = float(
    np.quantile(
        ref_price,
        0.995
    )
)

ref_price_clean = np.clip(
    ref_price,
    0,
    ref_price_upper
)

ref_price_sorted = np.sort(ref_price_clean)

# ------------------------------------------------------------
# Reference: customer transaction behavior
# ------------------------------------------------------------

ref_order_headers = (
    reference_retail
    .groupby(
        ["Invoice", "CustomerID"],
        as_index=False
    )
    .agg(
        InvoiceDate=("InvoiceDate", "min")
    )
)

ref_order_headers["Month"] = (
    ref_order_headers["InvoiceDate"]
    .dt.to_period("M")
)

ref_customer_monthly_orders = (
    ref_order_headers
    .groupby(
        ["CustomerID", "Month"]
    )
    .size()
    .astype(float)
)

REF_ACTIVE_MONTH_ORDER_MEAN = float(
    ref_customer_monthly_orders.mean()
)

REF_ACTIVE_MONTH_ORDER_MEDIAN = float(
    ref_customer_monthly_orders.median()
)

ref_customer_order_count = (
    ref_order_headers
    .groupby("CustomerID")
    .size()
    .astype(float)
)

# ------------------------------------------------------------
# Reference: purchase interval
# ------------------------------------------------------------

ref_order_headers = ref_order_headers.sort_values(
    ["CustomerID", "InvoiceDate"]
)

ref_order_headers["IntervalDays"] = (
    ref_order_headers
    .groupby("CustomerID")[
        "InvoiceDate"
    ]
    .diff()
    .dt.days
)

ref_purchase_intervals = (
    ref_order_headers[
        "IntervalDays"
    ]
    .dropna()
    .loc[
        lambda s: s.ge(0)
    ]
    .to_numpy(dtype=float)
)

REF_INTERVAL_MEDIAN = float(
    np.median(
        ref_purchase_intervals
    )
)

# ------------------------------------------------------------
# Reference: Product popularity
# ------------------------------------------------------------

ref_product_popularity = (
    reference_retail
    .groupby("StockCode")[
        "Quantity"
    ]
    .sum()
    .loc[
        lambda s: s.gt(0)
    ]
    .sort_values(
        ascending=False
    )
)

ref_product_share = (
    ref_product_popularity
    / ref_product_popularity.sum()
).to_numpy(dtype=float)

# ------------------------------------------------------------
# Reference: Customer revenue concentration
# ------------------------------------------------------------

ref_customer_revenue = (
    reference_retail
    .groupby("CustomerID")[
        "Revenue"
    ]
    .sum()
    .loc[
        lambda s: s.gt(0)
    ]
    .sort_values(
        ascending=False
    )
)

ref_top20_n = max(
    1,
    int(
        np.ceil(
            len(ref_customer_revenue)
            * 0.20
        )
    )
)

REF_TOP20_REVENUE_SHARE = float(
    ref_customer_revenue
    .head(ref_top20_n)
    .sum()
    / ref_customer_revenue.sum()
)

# ------------------------------------------------------------
# MTA Calibration
# ------------------------------------------------------------

mta["Channel"] = (
    mta["Channel"]
    .astype(str)
    .str.strip()
)

mta["Campaign"] = (
    mta["Campaign"]
    .astype(str)
    .str.strip()
)

mta["Conversion"] = (
    mta["Conversion"]
    .astype(str)
    .str.strip()
    .str.lower()
)

mta["Converted"] = (
    mta["Conversion"]
    .isin(
        ["yes", "1", "true", "converted"]
    )
    .astype(int)
)

ref_channel_distribution = (
    mta["Channel"]
    .value_counts(
        normalize=True
    )
)

ref_channel_conversion = (
    mta
    .groupby("Channel")[
        "Converted"
    ]
    .mean()
)

ref_campaign_distribution = (
    mta.loc[
        ~mta["Campaign"].isin(
            ["-", "", "nan", "None"]
        ),
        "Campaign"
    ]
    .value_counts(
        normalize=True
    )
)

# ------------------------------------------------------------
# Map MTA channel -> Project channel
# ------------------------------------------------------------

CHANNEL_MAP = {
    "Email": "Email",
    "Search Ads": "Google",
    "Social Media": "Facebook",
    "Direct Traffic": "Organic",
    "Referral": "Referral",
    "Display Ads": "LINE"
}

project_channel_weights = {
    "Email": 0.0,
    "Google": 0.0,
    "Facebook": 0.0,
    "Organic": 0.0,
    "Referral": 0.0,
    "LINE": 0.0
}

for source_channel, probability in ref_channel_distribution.items():
    mapped = CHANNEL_MAP.get(
        source_channel,
        None
    )

    if mapped is not None:
        project_channel_weights[mapped] += float(
            probability
        )

# Referral 會由推薦關係直接決定，
# 非 Referral 客戶的 AcquisitionSource 要重新正規化。
non_referral_sources = [
    "Facebook",
    "Google",
    "LINE",
    "Email",
    "Organic"
]

non_referral_probs = np.array(
    [
        project_channel_weights[x]
        for x in non_referral_sources
    ],
    dtype=float
)

if non_referral_probs.sum() <= 0:
    non_referral_probs = np.array(
        [0.20, 0.25, 0.10, 0.10, 0.35],
        dtype=float
    )

non_referral_probs = (
    non_referral_probs
    / non_referral_probs.sum()
)

# 各 Project Channel 的相對轉換效果
channel_conversion_effect = {
    "Email": 1.0,
    "Google": 1.0,
    "Facebook": 1.0,
    "Organic": 1.0,
    "Referral": 1.0,
    "LINE": 1.0
}

for source_channel, conv in ref_channel_conversion.items():
    mapped = CHANNEL_MAP.get(
        source_channel,
        None
    )

    if mapped is not None:
        channel_conversion_effect[mapped] = float(
            conv
        )

mean_conversion = np.mean(
    list(
        channel_conversion_effect.values()
    )
)

for key in channel_conversion_effect:
    channel_conversion_effect[key] = (
        channel_conversion_effect[key]
        / mean_conversion
    )

# ------------------------------------------------------------
# Console calibration summary
# ------------------------------------------------------------

print()
print("-" * 78)
print("REFERENCE CALIBRATION SUMMARY")
print("-" * 78)

print(
    "Reference Lines / Order Mean  :",
    round(ref_lines_mean, 2)
)

print(
    "Reference Lines / Order Median:",
    round(ref_lines_median, 2)
)

print(
    "Reference Quantity Median     :",
    round(ref_quantity_median, 2)
)

print(
    "Reference Active-Month Orders :",
    round(REF_ACTIVE_MONTH_ORDER_MEAN, 2)
)

print(
    "Reference Purchase Gap Median :",
    round(REF_INTERVAL_MEDIAN, 2),
    "days"
)

print(
    "Reference Top20 Revenue Share :",
    f"{REF_TOP20_REVENUE_SHARE:.2%}"
)

print()
print("Reference Channel Distribution:")
for channel, probability in ref_channel_distribution.items():
    print(
        f"    {channel:<15}",
        f"{probability:.2%}"
    )

# ============================================================
# Sampling Functions derived from reference data
# ============================================================

def sample_reference_customer_intensity(size):
    """
    由真實 Customer transaction count 分布取得相對活躍度。
    使用 log1p 降低極端大客戶影響。
    """
    sampled = np.random.choice(
        ref_customer_order_count.to_numpy(),
        size=size,
        replace=True
    )

    log_sampled = np.log1p(
        sampled
    )

    low = float(
        np.quantile(
            log_sampled,
            0.05
        )
    )

    high = float(
        np.quantile(
            log_sampled,
            0.95
        )
    )

    if high <= low:
        return np.ones(size)

    normalized = (
        log_sampled - low
    ) / (
        high - low
    )

    normalized = np.clip(
        normalized,
        0,
        1
    )

    # 0.60 ~ 1.55 倍
    return (
        0.60
        + 0.95 * normalized
    )


def sample_reference_customer_value_factor(size):
    """
    從真實零售資料的 Customer Revenue 分布抽樣客戶價值因子。
    為避免單一極端批發客戶支配模擬資料，先截到 99.5 百分位，
    再以平均值正規化。此因子用來保留真實資料的長尾與 Pareto 特性。
    """
    source = ref_customer_revenue.to_numpy(dtype=float)
    upper = float(np.quantile(source, 0.995))
    source = np.clip(source, 1e-6, upper)

    sampled = np.random.choice(
        source,
        size=size,
        replace=True
    )

    sampled_mean = float(sampled.mean())

    if sampled_mean <= 0:
        return np.ones(size, dtype=float)

    factor = sampled / sampled_mean

    return np.clip(
        factor,
        0.08,
        12.0
    )


def sample_lines_per_order():
    """
    參考真實 Invoice line count 的分布形狀，
    再縮放到專題設定平均約 TARGET_AVG_LINES_PER_ORDER。
    """
    raw_lines = float(
        np.random.choice(
            ref_lines_per_order.to_numpy()
        )
    )

    scaled = (
        raw_lines
        / max(
            ref_lines_mean,
            1.0
        )
        * TARGET_AVG_LINES_PER_ORDER
    )

    # 加一點隨機波動，避免只形成固定離散映射
    scaled *= np.random.uniform(
        0.85,
        1.15
    )

    return int(
        np.clip(
            round(scaled),
            1,
            MAX_LINES_PER_ORDER
        )
    )


def sample_quantity():
    """
    參考真實 Quantity 分布，但以中位數正規化，
    映射成適合本專題的 1~8 件。
    """
    raw_qty = float(
        np.random.choice(
            ref_quantity_clean
        )
    )

    ratio = (
        raw_qty
        / ref_quantity_median
    )

    quantity = int(
        round(
            1
            + np.log1p(
                max(
                    ratio - 1,
                    0
                )
            )
        )
    )

    return int(
        np.clip(
            quantity,
            1,
            MAX_QUANTITY_PER_LINE
        )
    )


def sample_relative_price_position():
    """
    從真實價格分布抽樣，轉換成 0~1 相對價格位置。
    使用排序後陣列 + searchsorted，避免每產生一個商品都掃描 120 萬筆資料。
    """
    raw_price = float(
        np.random.choice(
            ref_price_clean
        )
    )

    percentile = (
        np.searchsorted(
            ref_price_sorted,
            raw_price,
            side="right"
        )
        / len(ref_price_sorted)
    )

    return float(
        np.clip(
            percentile,
            0.02,
            0.98
        )
    )


# ============================================================
# 1. Customer_Dim
# ============================================================

print("\n[1/8] Generating Customer_Dim...")

n_2023 = int(
    N_CUSTOMERS * 0.20
)

n_2024 = int(
    N_CUSTOMERS * 0.35
)

n_2025 = (
    N_CUSTOMERS
    - n_2023
    - n_2024
)

register_dates = np.concatenate([
    np.random.choice(
        pd.date_range(
            "2023-01-01",
            "2023-12-31",
            freq="D"
        ),
        n_2023,
        replace=True
    ),

    np.random.choice(
        pd.date_range(
            "2024-01-01",
            "2024-12-31",
            freq="D"
        ),
        n_2024,
        replace=True
    ),

    np.random.choice(
        pd.date_range(
            "2025-01-01",
            "2025-12-31",
            freq="D"
        ),
        n_2025,
        replace=True
    )
])

# DatetimeIndex 沒有 .sort()，改成 numpy sort
register_dates = np.sort(
    pd.to_datetime(
        register_dates
    ).to_numpy()
)

customers = pd.DataFrame({
    "CustomerID": np.arange(
        1,
        N_CUSTOMERS + 1
    ),

    "Gender": np.random.choice(
        ["Male", "Female"],
        N_CUSTOMERS,
        p=[0.50, 0.50]
    ),

    "Age": np.random.randint(
        18,
        70,
        N_CUSTOMERS
    ),

    "City": np.random.choice(
        [
            "Taipei",
            "Taichung",
            "Tainan",
            "Kaohsiung",
            "Hsinchu"
        ],
        N_CUSTOMERS
    ),

    "RegisterDate": register_dates
})

# ============================================================
# 2. Hidden Behavior Factors
# ============================================================

print("[2/8] Generating hidden behavior factors...")

reference_intensity = sample_reference_customer_intensity(
    N_CUSTOMERS
)

reference_value_factor = sample_reference_customer_value_factor(
    N_CUSTOMERS
)

customers["BehaviorType"] = np.random.choice(
    [
        "Loyal",
        "Regular",
        "Growth",
        "ChurnProne"
    ],
    N_CUSTOMERS,
    p=[
        0.20,
        0.45,
        0.15,
        0.20
    ]
)

# 原始規則 Beta(4,3)，再加入真實客戶活躍度
base_retention = np.random.beta(
    4,
    3,
    N_CUSTOMERS
)

# 原始規則 Uniform(0.02,0.08)
decay_factor = np.random.uniform(
    0.02,
    0.08,
    N_CUSTOMERS
)

# 真實資料 active month 約 1.44 張訂單/月，
# 用真實值作為基準，再依客戶 intensity 放大/縮小。
purchase_frequency = (
    REF_ACTIVE_MONTH_ORDER_MEAN
    * reference_intensity
    * np.power(
        reference_value_factor,
        0.68
    )
)

basket_factor = np.random.lognormal(
    mean=0,
    sigma=0.50,
    size=N_CUSTOMERS
)

campaign_sensitivity = np.random.beta(
    2,
    2,
    N_CUSTOMERS
)

price_sensitivity = np.random.beta(
    2,
    2,
    N_CUSTOMERS
)

competitor_risk = np.random.beta(
    2,
    5,
    N_CUSTOMERS
)

referral_probability = np.random.beta(
    1,
    4,
    N_CUSTOMERS
)

# 隱藏型態只是調節，不直接輸出
retention_factor = customers[
    "BehaviorType"
].map({
    "Loyal": 1.18,
    "Regular": 1.00,
    "Growth": 0.95,
    "ChurnProne": 0.88
}).to_numpy()

frequency_factor = customers[
    "BehaviorType"
].map({
    "Loyal": 1.30,
    "Regular": 1.00,
    "Growth": 0.90,
    "ChurnProne": 0.90
}).to_numpy()

decay_type_factor = customers[
    "BehaviorType"
].map({
    "Loyal": 0.65,
    "Regular": 0.90,
    "Growth": 0.80,
    "ChurnProne": 1.30
}).to_numpy()

customers["BaseRetention"] = np.clip(
    base_retention
    * retention_factor
    * (
        0.85
        + 0.20
        * reference_intensity
    ),
    0.15,
    0.97
)

customers["DecayFactor"] = np.clip(
    decay_factor
    * decay_type_factor,
    0.005,
    0.12
)

customers["PurchaseFrequency"] = np.clip(
    purchase_frequency
    * frequency_factor,
    0.10,
    8.00
)

customers["BasketFactor"] = np.clip(
    basket_factor
    * (
        0.85
        + 0.20
        * reference_intensity
    )
    * np.power(
        reference_value_factor,
        0.24
    ),
    0.35,
    4.50
)

customers["CampaignSensitivity"] = (
    campaign_sensitivity
)

customers["PriceSensitivity"] = (
    price_sensitivity
)

customers["CompetitorRisk"] = (
    competitor_risk
)

customers["ReferralProbability"] = (
    referral_probability
)

# ------------------------------------------------------------
# True Churn State
# ------------------------------------------------------------

eligible_mask = (
    customers["RegisterDate"]
    <= OBSERVATION_END
)

eligible_indices = customers.index[
    eligible_mask
].to_numpy()

eligible_data = customers.loc[
    eligible_mask
].copy()

churn_propensity = (
    (
        1
        - eligible_data[
            "BaseRetention"
        ].to_numpy()
    )
    * 0.35
    +
    (
        eligible_data[
            "DecayFactor"
        ].to_numpy()
        / 0.12
    )
    * 0.25
    +
    eligible_data[
        "CompetitorRisk"
    ].to_numpy()
    * 0.30
    +
    (
        1
        / (
            1
            + eligible_data[
                "PurchaseFrequency"
            ].to_numpy()
        )
    )
    * 0.10
)

churn_propensity *= np.where(
    eligible_data[
        "BehaviorType"
    ].eq(
        "ChurnProne"
    ),
    1.45,
    1.00
)

churn_propensity = np.clip(
    churn_propensity,
    1e-6,
    None
)

churn_weights = (
    churn_propensity
    / churn_propensity.sum()
)

n_target_churn = int(
    round(
        len(
            eligible_indices
        )
        * TARGET_CHURN_RATE
    )
)

target_churn_indices = np.random.choice(
    eligible_indices,
    size=n_target_churn,
    replace=False,
    p=churn_weights
)

customers["TrueChurnState"] = False

customers.loc[
    target_churn_indices,
    "TrueChurnState"
] = True

customers["ChurnStartDate"] = pd.NaT

for idx in target_churn_indices:
    register_date = pd.Timestamp(
        customers.at[
            idx,
            "RegisterDate"
        ]
    )

    churn_window_start = max(
        pd.Timestamp(
            "2024-07-01"
        ),
        register_date
        + pd.Timedelta(
            days=60
        )
    )

    churn_window_end = pd.Timestamp(
        "2024-12-31"
    )

    if (
        churn_window_start
        > churn_window_end
    ):
        churn_date = pd.Timestamp(
            "2024-12-31"
        )
    else:
        churn_date = random_date(
            churn_window_start,
            churn_window_end
        )

    customers.at[
        idx,
        "ChurnStartDate"
    ] = churn_date

# ============================================================
# 3. Referral Relationship
# ============================================================

print("[3/8] Generating referral relationships...")

customers["ReferralUserID"] = 0
customers["ReferralDate"] = pd.NaT

register_array = customers[
    "RegisterDate"
].to_numpy(
    dtype="datetime64[ns]"
)

eligible_referral_targets = []

for idx in range(
    1,
    N_CUSTOMERS
):
    target_date = pd.Timestamp(
        register_array[idx]
    )

    min_referrer_date = np.datetime64(
        (
            target_date
            - pd.Timedelta(
                days=90
            )
        ).to_datetime64()
    )

    max_referrer_date = np.datetime64(
        (
            target_date
            - pd.Timedelta(
                days=1
            )
        ).to_datetime64()
    )

    previous_dates = register_array[
        :idx
    ]

    left = np.searchsorted(
        previous_dates,
        min_referrer_date,
        side="left"
    )

    right = np.searchsorted(
        previous_dates,
        max_referrer_date,
        side="right"
    )

    if right > left:
        eligible_referral_targets.append(
            (
                idx,
                left,
                right
            )
        )

target_referral_count = min(
    int(
        round(
            N_CUSTOMERS
            * REFERRAL_RATE
        )
    ),
    len(
        eligible_referral_targets
    )
)

selected_referral_positions = (
    np.random.choice(
        len(
            eligible_referral_targets
        ),
        size=target_referral_count,
        replace=False
    )
)

for position in selected_referral_positions:
    target_idx, left, right = (
        eligible_referral_targets[
            position
        ]
    )

    candidate_indices = np.arange(
        left,
        right
    )

    candidate_referral_weights = customers.loc[
        candidate_indices,
        "ReferralProbability"
    ].to_numpy(
        dtype=float
    )

    if (
        candidate_referral_weights.sum()
        <= 0
    ):
        candidate_referral_weights = (
            np.ones(
                len(
                    candidate_indices
                )
            )
        )

    candidate_referral_weights = (
        candidate_referral_weights
        / candidate_referral_weights.sum()
    )

    referrer_idx = int(
        np.random.choice(
            candidate_indices,
            p=candidate_referral_weights
        )
    )

    referrer_customer_id = int(
        customers.at[
            referrer_idx,
            "CustomerID"
        ]
    )

    referrer_register_date = pd.Timestamp(
        customers.at[
            referrer_idx,
            "RegisterDate"
        ]
    )

    new_register_date = pd.Timestamp(
        customers.at[
            target_idx,
            "RegisterDate"
        ]
    )

    # ReferralDate 位於推薦者註冊後、被推薦人註冊前
    referral_date = random_date(
        referrer_register_date,
        new_register_date
    )

    customers.at[
        target_idx,
        "ReferralUserID"
    ] = referrer_customer_id

    customers.at[
        target_idx,
        "ReferralDate"
    ] = referral_date

customers["ReferralUserID"] = customers[
    "ReferralUserID"
].astype(int)

customers["AcquisitionSource"] = np.random.choice(
    non_referral_sources,
    size=N_CUSTOMERS,
    p=non_referral_probs
)

customers.loc[
    customers[
        "ReferralUserID"
    ].ne(0),
    "AcquisitionSource"
] = "Referral"

customer_output_columns = [
    "CustomerID",
    "Gender",
    "Age",
    "City",
    "RegisterDate",
    "ReferralUserID",
    "AcquisitionSource"
]

customers[
    customer_output_columns
].to_csv(
    OUTPUT_DIR
    / "Customer_Dim.csv",
    index=False
)

# ============================================================
# 4. Product_Dim
# ============================================================

print("[4/8] Generating Product_Dim...")

product_categories = np.random.choice(
    [
        "Fresh Food",
        "Drink",
        "Snack",
        "Frozen",
        "Daily Goods"
    ],
    N_PRODUCTS,
    p=[
        0.25,
        0.20,
        0.20,
        0.15,
        0.20
    ]
)

products = pd.DataFrame({
    "ProductID": np.arange(
        1,
        N_PRODUCTS + 1
    ),

    "Category": product_categories,

    "Brand": np.random.choice(
        [
            "BrandA",
            "BrandB",
            "BrandC",
            "BrandD",
            "BrandE"
        ],
        N_PRODUCTS
    )
})

price_ranges = {
    "Fresh Food": (
        50,
        180
    ),
    "Drink": (
        20,
        80
    ),
    "Snack": (
        20,
        120
    ),
    "Frozen": (
        80,
        300
    ),
    "Daily Goods": (
        50,
        500
    )
}

generated_prices = []

for category in products[
    "Category"
]:
    low, high = price_ranges[
        category
    ]

    position = (
        sample_relative_price_position()
    )

    price = (
        low
        + position
        * (
            high
            - low
        )
    )

    generated_prices.append(
        int(
            round(
                price
            )
        )
    )

products["Price"] = generated_prices

products["Cost"] = (
    products["Price"]
    * np.random.uniform(
        0.40,
        0.75,
        N_PRODUCTS
    )
).round(0)

# ------------------------------------------------------------
# Product popularity
# 直接從真實商品銷售權重分布取樣，再正規化。
# ------------------------------------------------------------

sampled_popularity = np.random.choice(
    ref_product_share,
    size=N_PRODUCTS,
    replace=True
)

# 讓數值保持正值
sampled_popularity = np.clip(
    sampled_popularity,
    1e-12,
    None
)

products["PopularityWeight"] = (
    sampled_popularity
    / sampled_popularity.sum()
)

products["LaunchDate"] = np.random.choice(
    pd.date_range(
        "2022-01-01",
        "2025-06-30",
        freq="D"
    ),
    N_PRODUCTS
)

products["ShelfLifeDays"] = np.where(
    products[
        "Category"
    ].eq(
        "Fresh Food"
    ),
    np.random.randint(
        1,
        6,
        N_PRODUCTS
    ),
    np.random.randint(
        30,
        366,
        N_PRODUCTS
    )
)

products.to_csv(
    OUTPUT_DIR
    / "Product_Dim.csv",
    index=False
)

# ============================================================
# 5. Marketing_Campaign_Table
# ============================================================

print("[5/8] Generating Marketing_Campaign_Table...")

mta_campaign_names = (
    ref_campaign_distribution
    .index
    .tolist()
)

if not mta_campaign_names:
    mta_campaign_names = [
        "Retargeting",
        "Brand Awareness",
        "Discount Offer",
        "New Product Launch",
        "Seasonal Sale"
    ]

campaign_channel_probs = np.array([
    project_channel_weights["Facebook"],
    project_channel_weights["Google"],
    project_channel_weights["LINE"],
    project_channel_weights["Email"]
], dtype=float)

if campaign_channel_probs.sum() <= 0:
    campaign_channel_probs = np.ones(4, dtype=float)

campaign_channel_probs = (
    campaign_channel_probs
    / campaign_channel_probs.sum()
)

campaign_channels = np.random.choice(
    [
        "Facebook",
        "Google",
        "LINE",
        "Email"
    ],
    size=N_CAMPAIGNS,
    p=campaign_channel_probs
)

campaign = pd.DataFrame({
    "CampaignID": np.arange(
        1,
        N_CAMPAIGNS + 1
    ),

    "CampaignName": [
        f"{np.random.choice(mta_campaign_names)}_{i}"
        for i in range(
            1,
            N_CAMPAIGNS + 1
        )
    ],

    "Channel": campaign_channels,

    "Budget": np.random.choice(
        np.arange(
            5,
            51
        )
        * 10_000,
        N_CAMPAIGNS
    )
})

campaign_start_pool = pd.date_range(
    START_TS,
    END_TS
    - pd.Timedelta(
        days=30
    ),
    freq="D"
)

campaign["StartDate"] = np.sort(
    np.random.choice(
        campaign_start_pool,
        size=N_CAMPAIGNS,
        replace=False
    )
)

campaign_duration = np.random.randint(
    30,
    91,
    N_CAMPAIGNS
)

campaign["EndDate"] = (
    campaign["StartDate"]
    + pd.to_timedelta(
        campaign_duration
        - 1,
        unit="D"
    )
)

campaign["EndDate"] = campaign[
    "EndDate"
].clip(
    upper=END_TS
)

campaign["DiscountRate"] = np.random.choice(
    [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25
    ],
    N_CAMPAIGNS,
    p=[
        0.20,
        0.30,
        0.25,
        0.15,
        0.10
    ]
)

campaign["CampaignType"] = np.random.choice(
    [
        "Coupon",
        "Bundle",
        "Discount",
        "MemberDay",
        "Festival"
    ],
    N_CAMPAIGNS
)

campaign.to_csv(
    OUTPUT_DIR
    / "Marketing_Campaign_Table.csv",
    index=False
)

# ============================================================
# Lookups
# ============================================================

product_ids = products[
    "ProductID"
].to_numpy()

product_price = products.set_index(
    "ProductID"
)["Price"].to_dict()

product_category = products.set_index(
    "ProductID"
)["Category"].to_dict()

product_launch = (
    pd.to_datetime(
        products.set_index(
            "ProductID"
        )[
            "LaunchDate"
        ]
    ).to_dict()
)

product_weight = products.set_index(
    "ProductID"
)["PopularityWeight"].to_dict()

drink_ids = products.loc[
    products[
        "Category"
    ].eq(
        "Drink"
    ),
    "ProductID"
].to_numpy()

campaigns_by_date = {}

for row in campaign.itertuples(
    index=False
):
    for dt in pd.date_range(
        row.StartDate,
        row.EndDate,
        freq="D"
    ):
        campaigns_by_date.setdefault(
            pd.Timestamp(
                dt
            ).normalize(),
            []
        ).append(
            (
                int(
                    row.CampaignID
                ),
                float(
                    row.DiscountRate
                ),
                row.Channel
            )
        )


def choose_products_for_order(
    order_date,
    basket_factor
):
    """
    使用真實 Invoice line count 的分布形狀決定商品種類數，
    並依真實商品 popularity 形狀抽商品。
    """
    order_date = pd.Timestamp(
        order_date
    ).normalize()

    available_ids = [
        int(pid)
        for pid in product_ids
        if (
            pd.Timestamp(
                product_launch[
                    int(pid)
                ]
            ).normalize()
            <= order_date
        )
    ]

    if not available_ids:
        available_ids = [
            int(pid)
            for pid in product_ids
        ]

    base_lines = sample_lines_per_order()

    basket_multiplier = float(
        np.clip(
            basket_factor,
            0.50,
            2.50
        )
        ** 0.45
    )

    n_lines = int(
        np.clip(
            round(
                base_lines
                * basket_multiplier
            ),
            1,
            min(
                MAX_LINES_PER_ORDER,
                len(available_ids)
            )
        )
    )

    weights = np.array(
        [
            product_weight[
                pid
            ]
            for pid in available_ids
        ],
        dtype=float
    )

    weights = weights / weights.sum()

    chosen = np.random.choice(
        available_ids,
        size=n_lines,
        replace=False,
        p=weights
    ).astype(int).tolist()

    # Market Basket:
    # Fresh Food 60% 機率搭 Drink
    chosen_categories = {
        product_category[
            pid
        ]
        for pid in chosen
    }

    if (
        "Fresh Food"
        in chosen_categories
        and
        "Drink"
        not in chosen_categories
        and
        np.random.rand()
        < 0.60
    ):
        available_drinks = [
            int(pid)
            for pid in drink_ids
            if (
                pd.Timestamp(
                    product_launch[
                        int(pid)
                    ]
                ).normalize()
                <= order_date
            )
        ]

        if available_drinks:
            drink_pid = int(
                np.random.choice(
                    available_drinks
                )
            )

            if (
                drink_pid
                not in chosen
                and len(chosen)
                < MAX_LINES_PER_ORDER
            ):
                chosen.append(
                    drink_pid
                )

    return chosen


def append_order(
    order_rows,
    current_order_id,
    customer,
    order_date
):
    """
    一張訂單可產生多筆商品明細。
    """
    customer_id = int(
        customer.CustomerID
    )

    order_date = pd.Timestamp(
        order_date
    ).normalize()

    active_campaigns = campaigns_by_date.get(
        order_date,
        []
    )

    campaign_id = 0
    discount_rate = 0.0

    if active_campaigns:
        # 參考 MTA channel 的相對 conversion effect
        campaign_response_scores = []

        for (
            cid,
            rate,
            channel
        ) in active_campaigns:
            channel_effect = channel_conversion_effect.get(
                channel,
                1.0
            )

            score = (
                float(
                    customer.CampaignSensitivity
                )
                * (
                    0.50
                    + 0.50
                    * float(
                        customer.PriceSensitivity
                    )
                )
                * channel_effect
            )

            campaign_response_scores.append(
                max(
                    score,
                    1e-6
                )
            )

        overall_response_probability = safe_probability(
            0.12
            + 0.55
            * np.mean(
                campaign_response_scores
            )
        )

        if (
            np.random.rand()
            < overall_response_probability
        ):
            selected_idx = int(
                np.random.choice(
                    len(
                        active_campaigns
                    ),
                    p=(
                        np.asarray(
                            campaign_response_scores,
                            dtype=float
                        )
                        /
                        np.sum(
                            campaign_response_scores
                        )
                    )
                )
            )

            (
                campaign_id,
                discount_rate,
                _
            ) = active_campaigns[
                selected_idx
            ]

    channel = np.random.choice(
        [
            "APP",
            "WEB",
            "STORE"
        ],
        p=[
            0.35,
            0.25,
            0.40
        ]
    )

    store_id = int(
        np.random.randint(
            1,
            201
        )
    )

    selected_products = (
        choose_products_for_order(
            order_date,
            float(customer.BasketFactor)
        )
    )

    for product_id in selected_products:
        unit_price = float(
            product_price[
                product_id
            ]
        )

        quantity = sample_quantity()

        # BasketFactor 同時影響同品項購買件數，但維持上限避免極端值。
        quantity = int(
            np.clip(
                round(
                    quantity
                    * (
                        np.clip(
                            float(customer.BasketFactor),
                            0.50,
                            2.50
                        )
                        ** 0.30
                    )
                ),
                1,
                MAX_QUANTITY_PER_LINE
            )
        )

        gross_amount = (
            unit_price
            * quantity
        )

        discount_amount = (
            gross_amount
            * discount_rate
        )

        amount = (
            gross_amount
            - discount_amount
        )

        order_rows.append({
            "OrderID": current_order_id,
            "CustomerID": customer_id,
            "ProductID": int(
                product_id
            ),
            "OrderDate": order_date,
            "Quantity": int(
                quantity
            ),
            "UnitPrice": round(
                unit_price,
                2
            ),
            "GrossAmount": round(
                gross_amount,
                2
            ),
            "DiscountRate": round(
                discount_rate,
                4
            ),
            "DiscountAmount": round(
                discount_amount,
                2
            ),
            "Amount": round(
                amount,
                2
            ),
            "Channel": channel,
            "StoreID": store_id,
            "CampaignID": int(
                campaign_id
            )
        })

    return (
        current_order_id
        + 1
    )


# ============================================================
# 6. Order_Fact
# 一列 = 一個訂單商品明細
# ============================================================

print("[6/8] Generating Order_Fact...")

orders = []
order_id = 1

for customer_no, customer in enumerate(
    customers.itertuples(
        index=False
    ),
    start=1
):
    customer_id = int(
        customer.CustomerID
    )

    register_date = pd.Timestamp(
        customer.RegisterDate
    ).normalize()

    if (
        register_date
        > END_TS
    ):
        continue

    churn_start_date = None

    if pd.notna(
        customer.ChurnStartDate
    ):
        churn_start_date = pd.Timestamp(
            customer.ChurnStartDate
        ).normalize()

    first_order_date = (
        register_date
        + pd.Timedelta(
            days=np.random.randint(
                0,
                31
            )
        )
    )

    first_order_date = min(
        first_order_date,
        END_TS
    )

    current_month = (
        first_order_date
        .to_period("M")
        .to_timestamp()
    )

    month_index = 0

    while (
        current_month
        <= END_TS
    ):
        month_start = (
            current_month
        )

        month_end = min(
            current_month
            + pd.offsets.MonthEnd(
                0
            ),
            END_TS
        )

        # 真實流失後完全停止交易
        if (
            churn_start_date
            is not None
            and month_start
            > churn_start_date
        ):
            break

        retention = (
            float(
                customer.BaseRetention
            )
            * np.exp(
                -float(
                    customer.DecayFactor
                )
                * month_index
            )
            * (
                1
                - float(
                    customer.CompetitorRisk
                )
                * 0.20
            )
        )

        if (
            customer.BehaviorType
            == "Growth"
        ):
            retention *= min(
                1.25,
                1.0
                + 0.015
                * month_index
            )

        if (
            customer.BehaviorType
            == "Loyal"
        ):
            retention *= 1.08

        active_probability = float(
            np.clip(
                retention,
                0.03,
                0.95
            )
        )

        # 首月必須能形成首購
        is_active = (
            month_index == 0
            or
            np.random.rand()
            < active_probability
        )

        if is_active:
            usable_start = max(
                month_start,
                register_date,
                (
                    first_order_date
                    if month_index == 0
                    else month_start
                )
            )

            usable_end = (
                month_end
            )

            if (
                churn_start_date
                is not None
            ):
                usable_end = min(
                    usable_end,
                    churn_start_date
                )

            if (
                usable_start
                <= usable_end
            ):
                monthly_lambda = float(
                    customer.PurchaseFrequency
                )

                # Growth 類客戶隨時間略成長
                if (
                    customer.BehaviorType
                    == "Growth"
                ):
                    monthly_lambda *= min(
                        1.60,
                        1.0
                        + 0.025
                        * month_index
                    )

                # 當月 Campaign 存在時，提升交易頻率
                month_period = current_month.to_period(
                    "M"
                )

                has_campaign = False

                for campaign_row in campaign.itertuples(
                    index=False
                ):
                    if (
                        campaign_row.StartDate.to_period(
                            "M"
                        )
                        <= month_period
                        <= campaign_row.EndDate.to_period(
                            "M"
                        )
                    ):
                        has_campaign = True
                        break

                if has_campaign:
                    monthly_lambda *= (
                        1
                        + 0.18
                        * float(
                            customer.CampaignSensitivity
                        )
                    )

                monthly_transactions = max(
                    1,
                    int(
                        np.random.poisson(
                            max(
                                monthly_lambda,
                                0.15
                            )
                        )
                    )
                )

                monthly_transactions = min(
                    monthly_transactions,
                    10
                )

                available_days = (
                    usable_end.normalize()
                    - usable_start.normalize()
                ).days

                for _ in range(
                    monthly_transactions
                ):
                    order_date = (
                        usable_start.normalize()
                        + pd.Timedelta(
                            days=np.random.randint(
                                0,
                                available_days + 1
                            )
                        )
                    )

                    order_id = append_order(
                        order_rows=orders,
                        current_order_id=order_id,
                        customer=customer,
                        order_date=order_date
                    )

        current_month += pd.DateOffset(
            months=1
        )

        month_index += 1

    if (
        customer_no
        % 2_000
        == 0
    ):
        print(
            "    Customers processed:",
            f"{customer_no:,}/{N_CUSTOMERS:,}"
        )

if len(orders) == 0:
    raise ValueError(
        "Order_Fact 沒有產生任何訂單。"
    )

# ------------------------------------------------------------
# Ground Truth Calibration
#
# append_order() 的 order_rows 必須是 list。
# 因此校正期間只建立暫時 DataFrame 供查詢，
# 所有補單完成後才把 orders 正式轉為 DataFrame。
# ------------------------------------------------------------

orders_temp_df = pd.DataFrame(orders)
orders_temp_df["OrderDate"] = pd.to_datetime(
    orders_temp_df["OrderDate"]
)

future_customer_ids = set(
    orders_temp_df.loc[
        (
            orders_temp_df["OrderDate"]
            >= FUTURE_START
        )
        &
        (
            orders_temp_df["OrderDate"]
            <= FUTURE_END
        ),
        "CustomerID"
    ].astype(int)
)

non_churn_eligible = customers.loc[
    (
        customers["RegisterDate"]
        <= OBSERVATION_END
    )
    &
    (
        ~customers["TrueChurnState"]
    )
].copy()

missing_future = non_churn_eligible.loc[
    ~non_churn_eligible["CustomerID"].isin(
        future_customer_ids
    )
].copy()

print(
    "    Non-churn customers missing 2025 activity:",
    f"{len(missing_future):,}"
)

for customer in missing_future.itertuples(index=False):
    order_date = random_date(
        FUTURE_START,
        FUTURE_END
    )

    order_id = append_order(
        order_rows=orders,
        current_order_id=order_id,
        customer=customer,
        order_date=order_date
    )

# 所有訂單（含 2025 校正補單）完成後才轉 DataFrame
orders = pd.DataFrame(orders)
orders["OrderDate"] = pd.to_datetime(
    orders["OrderDate"]
)

orders = (
    orders
    .sort_values(
        [
            "OrderDate",
            "CustomerID",
            "OrderID",
            "ProductID"
        ]
    )
    .reset_index(
        drop=True
    )
)

orders.to_csv(
    OUTPUT_DIR
    / "Order_Fact.csv",
    index=False
)

print(
    "    Order Line Rows:",
    f"{len(orders):,}"
)

print(
    "    Transactions:",
    f"{orders['OrderID'].nunique():,}"
)

print(
    "    Unique Customers:",
    f"{orders['CustomerID'].nunique():,}"
)

# ============================================================
# 7. Event_Fact
# ============================================================

print("[7/8] Generating Event_Fact...")

events = []
event_id = 1

customer_register_lookup = (
    customers.set_index(
        "CustomerID"
    )[
        "RegisterDate"
    ].to_dict()
)

customer_churn_lookup = (
    customers.set_index(
        "CustomerID"
    )[
        "ChurnStartDate"
    ].to_dict()
)

customer_frequency_lookup = (
    customers.set_index(
        "CustomerID"
    )[
        "PurchaseFrequency"
    ].to_dict()
)

# ------------------------------------------------------------
# Register
# ------------------------------------------------------------

for customer in customers.itertuples(
    index=False
):
    events.append({
        "EventID": event_id,
        "CustomerID": int(
            customer.CustomerID
        ),
        "EventDate": pd.Timestamp(
            customer.RegisterDate
        ),
        "EventType": "Register",
        "Channel": np.random.choice(
            [
                "APP",
                "WEB"
            ]
        ),
        "SessionID": np.random.randint(
            100_000,
            1_000_000
        )
    })

    event_id += 1

# ------------------------------------------------------------
# Order-driven funnel
# 每張 OrderID 只建立一筆 Purchase Event
# ------------------------------------------------------------

order_headers = (
    orders
    .groupby(
        [
            "OrderID",
            "CustomerID",
            "OrderDate"
        ],
        as_index=False
    )
    .agg(
        Channel=(
            "Channel",
            "first"
        )
    )
)

for row_no, order in enumerate(
    order_headers.itertuples(
        index=False
    ),
    start=1
):
    customer_id = int(
        order.CustomerID
    )

    order_date = pd.Timestamp(
        order.OrderDate
    ).normalize()

    register_date = pd.Timestamp(
        customer_register_lookup[
            customer_id
        ]
    ).normalize()

    order_channel = (
        order.Channel
    )

    digital_channel = (
        order_channel
        if order_channel in [
            "APP",
            "WEB"
        ]
        else np.random.choice(
            [
                "APP",
                "WEB"
            ]
        )
    )

    session_id = np.random.randint(
        100_000,
        1_000_000
    )

    # View: 0~7 days before
    view_date = max(
        register_date,
        order_date
        - pd.Timedelta(
            days=np.random.randint(
                0,
                8
            )
        )
    )

    events.append({
        "EventID": event_id,
        "CustomerID": customer_id,
        "EventDate": view_date,
        "EventType": "View",
        "Channel": digital_channel,
        "SessionID": session_id
    })

    event_id += 1

    # Search: 80%
    if (
        np.random.rand()
        < 0.80
    ):
        search_date = max(
            register_date,
            order_date
            - pd.Timedelta(
                days=np.random.randint(
                    0,
                    6
                )
            )
        )

        events.append({
            "EventID": event_id,
            "CustomerID": customer_id,
            "EventDate": search_date,
            "EventType": "Search",
            "Channel": digital_channel,
            "SessionID": session_id
        })

        event_id += 1

    # AddCart: 0~1 day before
    addcart_date = max(
        register_date,
        order_date
        - pd.Timedelta(
            days=np.random.randint(
                0,
                2
            )
        )
    )

    events.append({
        "EventID": event_id,
        "CustomerID": customer_id,
        "EventDate": addcart_date,
        "EventType": "AddCart",
        "Channel": digital_channel,
        "SessionID": session_id
    })

    event_id += 1

    # Purchase
    events.append({
        "EventID": event_id,
        "CustomerID": customer_id,
        "EventDate": order_date,
        "EventType": "Purchase",
        "Channel": order_channel,
        "SessionID": session_id
    })

    event_id += 1

    # Review: 15%
    if (
        np.random.rand()
        < 0.15
    ):
        review_date = (
            order_date
            + pd.Timedelta(
                days=np.random.randint(
                    1,
                    16
                )
            )
        )

        if (
            review_date
            <= END_TS
        ):
            events.append({
                "EventID": event_id,
                "CustomerID": customer_id,
                "EventDate": review_date,
                "EventType": "Review",
                "Channel": digital_channel,
                "SessionID": session_id
            })

            event_id += 1

    if (
        row_no
        % 100_000
        == 0
    ):
        print(
            "    Order events processed:",
            f"{row_no:,}/{len(order_headers):,}"
        )

# ------------------------------------------------------------
# Login Events
# ------------------------------------------------------------

for customer_no, customer in enumerate(
    customers.itertuples(
        index=False
    ),
    start=1
):
    register_date = pd.Timestamp(
        customer.RegisterDate
    ).normalize()

    login_end = END_TS

    if pd.notna(
        customer.ChurnStartDate
    ):
        login_end = min(
            login_end,
            pd.Timestamp(
                customer.ChurnStartDate
            ).normalize()
        )

    if (
        login_end
        < register_date
    ):
        continue

    current_month = (
        register_date
        .to_period("M")
        .to_timestamp()
    )

    while (
        current_month
        <= login_end
    ):
        month_start = max(
            current_month,
            register_date
        )

        month_end = min(
            current_month
            + pd.offsets.MonthEnd(
                0
            ),
            login_end
        )

        if (
            month_start
            <= month_end
        ):
            login_lambda = (
                0.75
                + 0.70
                * float(
                    customer.PurchaseFrequency
                )
            )

            login_count = int(
                np.random.poisson(
                    max(
                        login_lambda,
                        0.10
                    )
                )
            )

            available_days = (
                month_end.normalize()
                - month_start.normalize()
            ).days

            for _ in range(
                login_count
            ):
                login_date = (
                    month_start.normalize()
                    + pd.Timedelta(
                        days=np.random.randint(
                            0,
                            available_days
                            + 1
                        )
                    )
                )

                events.append({
                    "EventID": event_id,
                    "CustomerID": int(
                        customer.CustomerID
                    ),
                    "EventDate": login_date,
                    "EventType": "Login",
                    "Channel": np.random.choice(
                        [
                            "APP",
                            "WEB"
                        ]
                    ),
                    "SessionID": np.random.randint(
                        100_000,
                        1_000_000
                    )
                })

                event_id += 1

        current_month += pd.DateOffset(
            months=1
        )

# ============================================================
# 8. Referral_Table + Referral Events
# ============================================================

print("[8/8] Generating Referral_Table...")

referral = customers.loc[
    customers[
        "ReferralUserID"
    ].ne(0),
    [
        "CustomerID",
        "ReferralUserID",
        "ReferralDate"
    ]
].copy()

referral = referral.rename(
    columns={
        "CustomerID":
            "NewCustomerID",

        "ReferralUserID":
            "ReferrerCustomerID"
    }
)

referral = (
    referral
    .sort_values(
        [
            "ReferralDate",
            "ReferrerCustomerID",
            "NewCustomerID"
        ]
    )
    .reset_index(
        drop=True
    )
)

referral.insert(
    0,
    "ReferralID",
    np.arange(
        1,
        len(
            referral
        )
        + 1
    )
)

referral.to_csv(
    OUTPUT_DIR
    / "Referral_Table.csv",
    index=False
)

for row in referral.itertuples(
    index=False
):
    events.append({
        "EventID": event_id,
        "CustomerID": int(
            row.ReferrerCustomerID
        ),
        "EventDate": pd.Timestamp(
            row.ReferralDate
        ),
        "EventType": "Referral",
        "Channel": np.random.choice(
            [
                "APP",
                "WEB"
            ]
        ),
        "SessionID": np.random.randint(
            100_000,
            1_000_000
        )
    })

    event_id += 1

events = pd.DataFrame(
    events
)

events["EventDate"] = pd.to_datetime(
    events[
        "EventDate"
    ]
)

events = events.loc[
    (
        events[
            "EventDate"
        ]
        >= START_TS
    )
    &
    (
        events[
            "EventDate"
        ]
        <= END_TS
    )
].copy()

events = (
    events
    .sort_values(
        [
            "EventDate",
            "CustomerID",
            "EventType"
        ]
    )
    .reset_index(
        drop=True
    )
)

events["EventID"] = np.arange(
    1,
    len(
        events
    )
    + 1
)

events.to_csv(
    OUTPUT_DIR
    / "Event_Fact.csv",
    index=False
)

# ============================================================
# DATA VALIDATION
# ============================================================

print()
print("=" * 78)
print("DATA VALIDATION")
print("=" * 78)

print(
    "Customer Count:",
    f"{customers['CustomerID'].nunique():,}"
)

print(
    "Product Count:",
    f"{products['ProductID'].nunique():,}"
)

print(
    "Campaign Count:",
    f"{campaign['CampaignID'].nunique():,}"
)

print(
    "Order Line Count:",
    f"{len(orders):,}"
)

print(
    "Transaction Count:",
    f"{orders['OrderID'].nunique():,}"
)

print(
    "Event Count:",
    f"{len(events):,}"
)

print(
    "Referral Count:",
    f"{len(referral):,}"
)

print()
print(
    "Order Date Range:",
    orders[
        "OrderDate"
    ].min(),
    "~",
    orders[
        "OrderDate"
    ].max()
)

print(
    "Event Date Range:",
    events[
        "EventDate"
    ].min(),
    "~",
    events[
        "EventDate"
    ].max()
)

# ------------------------------------------------------------
# Order structure
# ------------------------------------------------------------

order_line_counts = (
    orders
    .groupby(
        "OrderID"
    )
    .size()
)

transaction_amount = (
    orders
    .groupby(
        "OrderID"
    )[
        "Amount"
    ]
    .sum()
)

print()
print(
    "Average Lines Per Order:",
    round(
        order_line_counts.mean(),
        2
    )
)

print(
    "Median Lines Per Order:",
    round(
        order_line_counts.median(),
        2
    )
)

print(
    "Average Order Amount:",
    round(
        transaction_amount.mean(),
        2
    )
)

print(
    "Average Line Amount:",
    round(
        orders[
            "Amount"
        ].mean(),
        2
    )
)

print(
    "Average Transactions Per Customer:",
    round(
        orders[
            "OrderID"
        ].nunique()
        /
        orders[
            "CustomerID"
        ].nunique(),
        2
    )
)

# ------------------------------------------------------------
# Referral
# ------------------------------------------------------------

actual_referral_rate = (
    customers[
        "ReferralUserID"
    ]
    .ne(0)
    .mean()
)

print()
print(
    "Referral Rate:",
    f"{actual_referral_rate:.2%}"
)

# ------------------------------------------------------------
# Revenue concentration
# ------------------------------------------------------------

customer_revenue = (
    orders
    .groupby(
        "CustomerID"
    )[
        "Amount"
    ]
    .sum()
    .sort_values(
        ascending=False
    )
)

top20_n = max(
    1,
    int(
        np.ceil(
            len(
                customer_revenue
            )
            * 0.20
        )
    )
)

top20_share = (
    customer_revenue
    .head(
        top20_n
    )
    .sum()
    /
    customer_revenue.sum()
)

print()
print(
    "Reference Top20 Revenue Share:",
    f"{REF_TOP20_REVENUE_SHARE:.2%}"
)

print(
    "Generated Top20 Revenue Share:",
    f"{top20_share:.2%}"
)

# ------------------------------------------------------------
# Campaign share
# ------------------------------------------------------------

campaign_transaction_share = (
    orders.loc[
        orders[
            "CampaignID"
        ].ne(0),
        "OrderID"
    ]
    .nunique()
    /
    orders[
        "OrderID"
    ]
    .nunique()
)

print(
    "Campaign Transaction Share:",
    f"{campaign_transaction_share:.2%}"
)

# ------------------------------------------------------------
# Churn Ground Truth
# ------------------------------------------------------------

eligible_customer_ids = set(
    customers.loc[
        customers[
            "RegisterDate"
        ]
        <= OBSERVATION_END,
        "CustomerID"
    ].astype(
        int
    )
)

future_customer_ids = set(
    orders.loc[
        (
            orders[
                "OrderDate"
            ]
            >= FUTURE_START
        )
        &
        (
            orders[
                "OrderDate"
            ]
            <= FUTURE_END
        ),
        "CustomerID"
    ].astype(
        int
    )
)

actual_churn_ids = (
    eligible_customer_ids
    - future_customer_ids
)

actual_churn_rate = (
    len(
        actual_churn_ids
    )
    /
    len(
        eligible_customer_ids
    )
)

print()
print(
    "Eligible Customers for Prediction:",
    f"{len(eligible_customer_ids):,}"
)

print(
    "2025 Actual Churn Customers:",
    f"{len(actual_churn_ids):,}"
)

print(
    "2025 Actual Churn Rate:",
    f"{actual_churn_rate:.2%}"
)

# ------------------------------------------------------------
# Future Revenue
# ------------------------------------------------------------

future_revenue = (
    orders.loc[
        (
            orders[
                "OrderDate"
            ]
            >= FUTURE_START
        )
        &
        (
            orders[
                "OrderDate"
            ]
            <= FUTURE_END
        )
        &
        (
            orders[
                "CustomerID"
            ].isin(
                eligible_customer_ids
            )
        )
    ]
    .groupby(
        "CustomerID"
    )[
        "Amount"
    ]
    .sum()
)

future_revenue_all = pd.Series(
    0.0,
    index=sorted(
        eligible_customer_ids
    )
)

future_revenue_all.loc[
    future_revenue.index
] = future_revenue.values

print()
print(
    "Future_12M_Revenue Zero Rate:",
    f"{(future_revenue_all == 0).mean():.2%}"
)

print(
    "Future_12M_Revenue Mean:",
    round(
        future_revenue_all.mean(),
        2
    )
)

print(
    "Future_12M_Revenue Median:",
    round(
        future_revenue_all.median(),
        2
    )
)

print(
    "Future_12M_Revenue Max:",
    round(
        future_revenue_all.max(),
        2
    )
)

# ------------------------------------------------------------
# Cohort Coverage
# ------------------------------------------------------------

first_purchase = (
    orders
    .groupby(
        "CustomerID"
    )[
        "OrderDate"
    ]
    .min()
    .dt.to_period(
        "M"
    )
)

expected_months = pd.period_range(
    "2023-01",
    "2025-12",
    freq="M"
)

actual_months = set(
    first_purchase.unique()
)

missing_cohort_months = [
    str(month)
    for month in expected_months
    if month
    not in actual_months
]

print()
print(
    "Missing Cohort Months:",
    (
        missing_cohort_months
        if missing_cohort_months
        else "None"
    )
)

# ------------------------------------------------------------
# Integrity
# ------------------------------------------------------------

register_map = (
    customers
    .set_index(
        "CustomerID"
    )[
        "RegisterDate"
    ]
)

order_register_date = (
    orders[
        "CustomerID"
    ].map(
        register_map
    )
)

event_register_date = (
    events[
        "CustomerID"
    ].map(
        register_map
    )
)

invalid_orders_before_register = int(
    (
        orders[
            "OrderDate"
        ]
        < order_register_date
    ).sum()
)

invalid_events_before_register = int(
    (
        events[
            "EventDate"
        ]
        < event_register_date
    ).sum()
)

invalid_quantity = int(
    (
        orders[
            "Quantity"
        ]
        < 1
    ).sum()
)

invalid_amount = int(
    (
        orders[
            "Amount"
        ]
        <= 0
    ).sum()
)

invalid_cost = int(
    (
        products[
            "Cost"
        ]
        >= products[
            "Price"
        ]
    ).sum()
)

print()
print(
    "Orders Before RegisterDate:",
    invalid_orders_before_register
)

print(
    "Events Before RegisterDate:",
    invalid_events_before_register
)

print(
    "Invalid Quantity Rows:",
    invalid_quantity
)

print(
    "Invalid Amount Rows:",
    invalid_amount
)

print(
    "Products Cost >= Price:",
    invalid_cost
)

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print()
print("=" * 78)
print("RAW DATA GENERATION FINISHED")
print("=" * 78)

print()
print("Output Files:")

for filename in [
    "Customer_Dim.csv",
    "Product_Dim.csv",
    "Order_Fact.csv",
    "Event_Fact.csv",
    "Referral_Table.csv",
    "Marketing_Campaign_Table.csv"
]:
    print(
        " -",
        OUTPUT_DIR
        / filename
    )
