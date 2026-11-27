import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ETL V1
# RAW_Data -> Cleaned ETL_Output
#
# 職責邊界：
# - 只做資料清理、型別標準化、完整性檢查、必要基礎欄位
# - 不做 RFM / Cohort / AARRR
# - 不做 ChurnLabel
# - 不做 Future_12M_Revenue
# - 不建立 Data Warehouse 維度/事實表
#
# 輸入：
#   原始資料/RAW_Data/
#
# 輸出：
#   ETL_Output/
#
# 來源六表：
# - Customer_Dim.csv
# - Product_Dim.csv
# - Order_Fact.csv
# - Event_Fact.csv
# - Referral_Table.csv
# - Marketing_Campaign_Table.csv
# ============================================================



# ============================================================
# Project Root Discovery
# ============================================================

def find_project_root():
    """
    從 ETL.py 所在位置一路往上尋找專題根目錄。

    判斷方式：
    1. 優先尋找存在「原始資料/RAW_Data」的父資料夾
    2. 若 ETL.py 本身放在「原始資料」附近，也能往上找到正確根目錄

    不使用固定 .parent.parent...。
    """
    current = Path(__file__).resolve().parent

    for parent in [current, *current.parents]:
        raw_candidate = parent / "原始資料" / "RAW_Data"
        if raw_candidate.is_dir():
            return parent

    # 兼容：若專題根目錄本身就是「原始資料」資料夾
    for parent in [current, *current.parents]:
        raw_candidate = parent / "RAW_Data"
        if raw_candidate.is_dir():
            return parent.parent

    raise FileNotFoundError(
        "找不到專題根目錄：未找到『原始資料/RAW_Data』資料夾。"
    )


PROJECT_DIR = find_project_root()
import sys
sys.path.insert(0,str(PROJECT_DIR))
from quarterly_plan import load_plan
PLAN = load_plan(PROJECT_DIR)
START_DATE = pd.Timestamp(PLAN.start_date)
END_DATE = pd.Timestamp(PLAN.score_end)
RAW_DIR = PROJECT_DIR / "原始資料" / "RAW_Data"
OUTPUT_DIR = PROJECT_DIR / "ETL_Output"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 86)
print("ETL V1 - RAW DATA CLEANING & STANDARDIZATION")
print("=" * 86)
print("Project Root :", PROJECT_DIR)
print("Raw Data Dir :", RAW_DIR)
print("ETL Output   :", OUTPUT_DIR)

# ============================================================
# Required Input Files
# ============================================================

REQUIRED_FILES = {
    "customer": "Customer_Dim.csv",
    "product": "Product_Dim.csv",
    "order": "Order_Fact.csv",
    "event": "Event_Fact.csv",
    "referral": "Referral_Table.csv",
    "campaign": "Marketing_Campaign_Table.csv",
}

for key, filename in REQUIRED_FILES.items():
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"找不到必要輸入檔案：{path}"
        )

# ============================================================
# Helpers
# ============================================================

def print_section(title):
    print()
    print("-" * 86)
    print(title)
    print("-" * 86)


def read_csv_checked(path, name):
    df = pd.read_csv(path, low_memory=False)
    print(f"{name:<30} rows = {len(df):,}")
    return df


def normalize_string_series(series):
    """
    一般字串清理：
    - 轉 pandas string dtype
    - 去除前後空白
    - 空字串 -> NA
    """
    s = series.astype("string").str.strip()
    s = s.replace("", pd.NA)
    return s


def convert_numeric(df, columns, integer=False):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
            if integer:
                # 尚未 dropna 前使用 nullable Int64
                df[col] = df[col].astype("Int64")
    return df


def convert_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )
    return df


def drop_full_duplicates(df, stats, table_name):
    before = len(df)
    df = df.drop_duplicates().copy()
    removed = before - len(df)
    stats[table_name]["full_duplicates_removed"] = removed
    return df


def require_columns(df, table_name, required_columns):
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            f"{table_name} 缺少必要欄位：{missing}"
        )


def finalize_int_columns(df, columns):
    """
    清理完成後，必要 ID 欄位改回 int64。
    """
    for col in columns:
        if col in df.columns:
            if df[col].isna().any():
                raise ValueError(
                    f"{col} 清理後仍存在空值，無法轉成 int64。"
                )
            df[col] = df[col].astype("int64")
    return df


def date_in_range(series):
    return series.between(
        START_DATE,
        END_DATE,
        inclusive="both"
    )


# ============================================================
# Load Raw Data
# ============================================================

print_section("1. LOADING RAW DATA")

customer = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["customer"],
    "Customer_Dim"
)

product = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["product"],
    "Product_Dim"
)

orders = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["order"],
    "Order_Fact"
)

events = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["event"],
    "Event_Fact"
)

referral = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["referral"],
    "Referral_Table"
)

campaign = read_csv_checked(
    RAW_DIR / REQUIRED_FILES["campaign"],
    "Marketing_Campaign_Table"
)

stats = {
    "Customer_Dim": {"input_rows": len(customer)},
    "Product_Dim": {"input_rows": len(product)},
    "Order_Fact": {"input_rows": len(orders)},
    "Event_Fact": {"input_rows": len(events)},
    "Referral_Table": {"input_rows": len(referral)},
    "Marketing_Campaign_Table": {"input_rows": len(campaign)},
}

# ============================================================
# Schema Validation
# ============================================================

print_section("2. SCHEMA VALIDATION")

require_columns(
    customer,
    "Customer_Dim",
    [
        "CustomerID",
        "Gender",
        "Age",
        "City",
        "RegisterDate",
        "ReferralUserID",
        "AcquisitionSource",
    ]
)

require_columns(
    product,
    "Product_Dim",
    [
        "ProductID",
        "Category",
        "Brand",
        "Price",
        "Cost",
        "PopularityWeight",
        "LaunchDate",
        "ShelfLifeDays",
    ]
)

require_columns(
    orders,
    "Order_Fact",
    [
        "OrderID",
        "CustomerID",
        "ProductID",
        "OrderDate",
        "Quantity",
        "UnitPrice",
        "GrossAmount",
        "DiscountRate",
        "DiscountAmount",
        "Amount",
        "Channel",
        "StoreID",
        "CampaignID",
    ]
)

require_columns(
    events,
    "Event_Fact",
    [
        "EventID",
        "CustomerID",
        "EventDate",
        "EventType",
        "Channel",
        "SessionID",
    ]
)

require_columns(
    referral,
    "Referral_Table",
    [
        "ReferralID",
        "ReferrerCustomerID",
        "NewCustomerID",
        "ReferralDate",
    ]
)

require_columns(
    campaign,
    "Marketing_Campaign_Table",
    [
        "CampaignID",
        "CampaignName",
        "Channel",
        "Budget",
        "StartDate",
        "EndDate",
        "DiscountRate",
        "CampaignType",
    ]
)

print("Schema validation: PASS")

# ============================================================
# 3. Customer_Dim Cleaning
# ============================================================

print_section("3. CLEANING Customer_Dim")

customer = drop_full_duplicates(
    customer,
    stats,
    "Customer_Dim"
)

customer = convert_numeric(
    customer,
    [
        "CustomerID",
        "Age",
        "ReferralUserID",
    ],
    integer=True
)

customer = convert_datetime(
    customer,
    ["RegisterDate"]
)

for col in [
    "Gender",
    "City",
    "AcquisitionSource",
]:
    customer[col] = normalize_string_series(
        customer[col]
    )

before = len(customer)

customer = customer.loc[
    customer["CustomerID"].notna()
    & customer["RegisterDate"].notna()
].copy()

stats["Customer_Dim"]["missing_required_removed"] = (
    before - len(customer)
)

# CustomerID 主鍵不可重複
before = len(customer)
customer = (
    customer
    .sort_values(
        ["CustomerID", "RegisterDate"]
    )
    .drop_duplicates(
        subset=["CustomerID"],
        keep="first"
    )
    .copy()
)
stats["Customer_Dim"]["duplicate_pk_removed"] = (
    before - len(customer)
)

# 基礎合法性
before = len(customer)

valid_customer_mask = (
    customer["Age"].between(18, 120, inclusive="both")
    & customer["RegisterDate"].le(END_DATE)
    & customer["Gender"].notna()
    & customer["City"].notna()
    & customer["AcquisitionSource"].notna()
)

customer = customer.loc[
    valid_customer_mask
].copy()

stats["Customer_Dim"]["invalid_rows_removed"] = (
    before - len(customer)
)

customer = finalize_int_columns(
    customer,
    [
        "CustomerID",
        "Age",
        "ReferralUserID",
    ]
)

# ============================================================
# 4. Product_Dim Cleaning
# ============================================================

print_section("4. CLEANING Product_Dim")

product = drop_full_duplicates(
    product,
    stats,
    "Product_Dim"
)

product = convert_numeric(
    product,
    [
        "ProductID",
        "Price",
        "Cost",
        "PopularityWeight",
        "ShelfLifeDays",
    ],
    integer=False
)

product["ProductID"] = pd.to_numeric(
    product["ProductID"],
    errors="coerce"
).astype("Int64")

product["ShelfLifeDays"] = pd.to_numeric(
    product["ShelfLifeDays"],
    errors="coerce"
).astype("Int64")

product = convert_datetime(
    product,
    ["LaunchDate"]
)

for col in [
    "Category",
    "Brand",
]:
    product[col] = normalize_string_series(
        product[col]
    )

before = len(product)

product = product.loc[
    product["ProductID"].notna()
    & product["Category"].notna()
    & product["Price"].notna()
    & product["Cost"].notna()
    & product["LaunchDate"].notna()
].copy()

stats["Product_Dim"]["missing_required_removed"] = (
    before - len(product)
)

before = len(product)

product = (
    product
    .sort_values(
        ["ProductID", "LaunchDate"]
    )
    .drop_duplicates(
        subset=["ProductID"],
        keep="first"
    )
    .copy()
)

stats["Product_Dim"]["duplicate_pk_removed"] = (
    before - len(product)
)

before = len(product)

valid_product_mask = (
    product["Price"].gt(0)
    & product["Cost"].ge(0)
    & product["Cost"].lt(product["Price"])
    & product["ShelfLifeDays"].gt(0)
    & product["PopularityWeight"].gt(0)
)

product = product.loc[
    valid_product_mask
].copy()

stats["Product_Dim"]["invalid_rows_removed"] = (
    before - len(product)
)

product = finalize_int_columns(
    product,
    [
        "ProductID",
        "ShelfLifeDays",
    ]
)

# ============================================================
# 5. Marketing_Campaign_Table Cleaning
# ============================================================

print_section("5. CLEANING Marketing_Campaign_Table")

campaign = drop_full_duplicates(
    campaign,
    stats,
    "Marketing_Campaign_Table"
)

campaign = convert_numeric(
    campaign,
    [
        "CampaignID",
        "Budget",
        "DiscountRate",
    ],
    integer=False
)

campaign["CampaignID"] = pd.to_numeric(
    campaign["CampaignID"],
    errors="coerce"
).astype("Int64")

campaign = convert_datetime(
    campaign,
    [
        "StartDate",
        "EndDate",
    ]
)

for col in [
    "CampaignName",
    "Channel",
    "CampaignType",
]:
    campaign[col] = normalize_string_series(
        campaign[col]
    )

before = len(campaign)

campaign = campaign.loc[
    campaign["CampaignID"].notna()
    & campaign["CampaignName"].notna()
    & campaign["Channel"].notna()
    & campaign["Budget"].notna()
    & campaign["StartDate"].notna()
    & campaign["EndDate"].notna()
].copy()

stats[
    "Marketing_Campaign_Table"
]["missing_required_removed"] = (
    before - len(campaign)
)

before = len(campaign)

campaign = (
    campaign
    .sort_values(
        ["CampaignID", "StartDate"]
    )
    .drop_duplicates(
        subset=["CampaignID"],
        keep="first"
    )
    .copy()
)

stats[
    "Marketing_Campaign_Table"
]["duplicate_pk_removed"] = (
    before - len(campaign)
)

before = len(campaign)

valid_campaign_mask = (
    campaign["Budget"].gt(0)
    & campaign["DiscountRate"].between(
        0,
        1,
        inclusive="both"
    )
    & campaign["EndDate"].ge(
        campaign["StartDate"]
    )
    & campaign["StartDate"].le(END_DATE)
    & campaign["EndDate"].ge(START_DATE)
)

campaign = campaign.loc[
    valid_campaign_mask
].copy()

stats[
    "Marketing_Campaign_Table"
]["invalid_rows_removed"] = (
    before - len(campaign)
)

campaign = finalize_int_columns(
    campaign,
    ["CampaignID"]
)

# ============================================================
# Lookup Sets
# ============================================================

valid_customer_ids = set(
    customer["CustomerID"].tolist()
)

valid_product_ids = set(
    product["ProductID"].tolist()
)

valid_campaign_ids = set(
    campaign["CampaignID"].tolist()
)

customer_register_map = (
    customer
    .set_index("CustomerID")["RegisterDate"]
)

product_launch_map = (
    product
    .set_index("ProductID")["LaunchDate"]
)

campaign_lookup = (
    campaign
    .set_index("CampaignID")[
        [
            "StartDate",
            "EndDate",
            "DiscountRate",
        ]
    ]
)

# ============================================================
# 6. Order_Fact Cleaning
# ============================================================

print_section("6. CLEANING Order_Fact")

orders = drop_full_duplicates(
    orders,
    stats,
    "Order_Fact"
)

orders = convert_numeric(
    orders,
    [
        "OrderID",
        "CustomerID",
        "ProductID",
        "Quantity",
        "UnitPrice",
        "GrossAmount",
        "DiscountRate",
        "DiscountAmount",
        "Amount",
        "StoreID",
        "CampaignID",
    ],
    integer=False
)

for col in [
    "OrderID",
    "CustomerID",
    "ProductID",
    "Quantity",
    "StoreID",
    "CampaignID",
]:
    orders[col] = pd.to_numeric(
        orders[col],
        errors="coerce"
    ).astype("Int64")

orders = convert_datetime(
    orders,
    ["OrderDate"]
)

orders["Channel"] = normalize_string_series(
    orders["Channel"]
)

before = len(orders)

orders = orders.loc[
    orders["OrderID"].notna()
    & orders["CustomerID"].notna()
    & orders["ProductID"].notna()
    & orders["OrderDate"].notna()
    & orders["Quantity"].notna()
    & orders["UnitPrice"].notna()
    & orders["GrossAmount"].notna()
    & orders["DiscountRate"].notna()
    & orders["DiscountAmount"].notna()
    & orders["Amount"].notna()
    & orders["StoreID"].notna()
    & orders["CampaignID"].notna()
].copy()

stats["Order_Fact"]["missing_required_removed"] = (
    before - len(orders)
)

# 外鍵檢查
before = len(orders)

orders = orders.loc[
    orders["CustomerID"].isin(valid_customer_ids)
    & orders["ProductID"].isin(valid_product_ids)
    & (
        orders["CampaignID"].eq(0)
        | orders["CampaignID"].isin(valid_campaign_ids)
    )
].copy()

stats["Order_Fact"]["fk_invalid_removed"] = (
    before - len(orders)
)

# 日期與商品上市日
orders["CustomerRegisterDate"] = (
    orders["CustomerID"]
    .map(customer_register_map)
)

orders["ProductLaunchDate"] = (
    orders["ProductID"]
    .map(product_launch_map)
)

before = len(orders)

valid_order_mask = (
    date_in_range(orders["OrderDate"])
    & orders["OrderDate"].ge(
        orders["CustomerRegisterDate"]
    )
    & orders["OrderDate"].ge(
        orders["ProductLaunchDate"]
    )
    & orders["Quantity"].gt(0)
    & orders["UnitPrice"].gt(0)
    & orders["GrossAmount"].gt(0)
    & orders["DiscountRate"].between(
        0,
        1,
        inclusive="both"
    )
    & orders["DiscountAmount"].ge(0)
    & orders["Amount"].gt(0)
)

orders = orders.loc[
    valid_order_mask
].copy()

stats["Order_Fact"]["invalid_rows_removed"] = (
    before - len(orders)
)

# ------------------------------------------------------------
# 金額一致性檢查
# 使用極小容忍值避免浮點誤差
# ------------------------------------------------------------

expected_gross = (
    orders["UnitPrice"]
    * orders["Quantity"]
)

expected_discount = (
    orders["GrossAmount"]
    * orders["DiscountRate"]
)

expected_amount = (
    orders["GrossAmount"]
    - orders["DiscountAmount"]
)

amount_tolerance = 0.02

before = len(orders)

consistent_amount_mask = (
    np.isclose(
        orders["GrossAmount"],
        expected_gross,
        atol=amount_tolerance,
        rtol=0
    )
    & np.isclose(
        orders["DiscountAmount"],
        expected_discount,
        atol=amount_tolerance,
        rtol=0
    )
    & np.isclose(
        orders["Amount"],
        expected_amount,
        atol=amount_tolerance,
        rtol=0
    )
)

orders = orders.loc[
    consistent_amount_mask
].copy()

stats["Order_Fact"]["amount_inconsistent_removed"] = (
    before - len(orders)
)

# ------------------------------------------------------------
# Campaign 邏輯檢查
# CampaignID=0 -> DiscountRate 必須為 0、DiscountAmount 必須為 0
# CampaignID>0 -> 訂單日期必須落在活動期間
#                  且折扣率與 Campaign Table 一致
# ------------------------------------------------------------

no_campaign_mask = orders["CampaignID"].eq(0)

before_no_campaign = len(orders)

valid_no_campaign = (
    ~no_campaign_mask
    | (
        orders["DiscountRate"].eq(0)
        & orders["DiscountAmount"].eq(0)
    )
)

orders = orders.loc[
    valid_no_campaign
].copy()

stats["Order_Fact"]["invalid_no_campaign_rows_removed"] = (
    before_no_campaign - len(orders)
)

campaign_orders = orders.loc[
    orders["CampaignID"].ne(0)
].copy()

if not campaign_orders.empty:
    campaign_orders["CampaignStartDate"] = (
        campaign_orders["CampaignID"]
        .map(campaign_lookup["StartDate"])
    )

    campaign_orders["CampaignEndDate"] = (
        campaign_orders["CampaignID"]
        .map(campaign_lookup["EndDate"])
    )

    campaign_orders["CampaignExpectedDiscountRate"] = (
        campaign_orders["CampaignID"]
        .map(campaign_lookup["DiscountRate"])
    )

    valid_campaign_order_mask = (
        campaign_orders["OrderDate"].between(
            campaign_orders["CampaignStartDate"],
            campaign_orders["CampaignEndDate"],
            inclusive="both"
        )
        & np.isclose(
            campaign_orders["DiscountRate"],
            campaign_orders["CampaignExpectedDiscountRate"],
            atol=1e-9,
            rtol=0
        )
    )

    valid_campaign_order_index = set(
        campaign_orders.loc[
            valid_campaign_order_mask
        ].index
    )

    before = len(orders)

    orders = orders.loc[
        orders["CampaignID"].eq(0)
        | orders.index.isin(valid_campaign_order_index)
    ].copy()

    stats["Order_Fact"]["invalid_campaign_rows_removed"] = (
        before - len(orders)
    )
else:
    stats["Order_Fact"]["invalid_campaign_rows_removed"] = 0

# ------------------------------------------------------------
# OrderID 一致性
#
# 一張訂單可以多 ProductID，
# 但同一 OrderID 的：
# CustomerID / OrderDate / Channel / StoreID / CampaignID
# 必須一致。
# ------------------------------------------------------------

order_header_nunique = (
    orders
    .groupby("OrderID")
    .agg(
        CustomerID_n=("CustomerID", "nunique"),
        OrderDate_n=("OrderDate", "nunique"),
        Channel_n=("Channel", "nunique"),
        StoreID_n=("StoreID", "nunique"),
        CampaignID_n=("CampaignID", "nunique"),
    )
)

invalid_order_ids = set(
    order_header_nunique.loc[
        (
            order_header_nunique["CustomerID_n"] > 1
        )
        | (
            order_header_nunique["OrderDate_n"] > 1
        )
        | (
            order_header_nunique["Channel_n"] > 1
        )
        | (
            order_header_nunique["StoreID_n"] > 1
        )
        | (
            order_header_nunique["CampaignID_n"] > 1
        )
    ].index
)

before = len(orders)

if invalid_order_ids:
    orders = orders.loc[
        ~orders["OrderID"].isin(
            invalid_order_ids
        )
    ].copy()

stats["Order_Fact"]["inconsistent_order_header_removed"] = (
    before - len(orders)
)

# 移除暫存欄位
orders = orders.drop(
    columns=[
        "CustomerRegisterDate",
        "ProductLaunchDate",
    ],
    errors="ignore"
)

orders = finalize_int_columns(
    orders,
    [
        "OrderID",
        "CustomerID",
        "ProductID",
        "Quantity",
        "StoreID",
        "CampaignID",
    ]
)

# ============================================================
# 7. Event_Fact Cleaning
# ============================================================

print_section("7. CLEANING Event_Fact")

events = drop_full_duplicates(
    events,
    stats,
    "Event_Fact"
)

events = convert_numeric(
    events,
    [
        "EventID",
        "CustomerID",
        "SessionID",
    ],
    integer=False
)

for col in [
    "EventID",
    "CustomerID",
    "SessionID",
]:
    events[col] = pd.to_numeric(
        events[col],
        errors="coerce"
    ).astype("Int64")

events = convert_datetime(
    events,
    ["EventDate"]
)

for col in [
    "EventType",
    "Channel",
]:
    events[col] = normalize_string_series(
        events[col]
    )

before = len(events)

events = events.loc[
    events["EventID"].notna()
    & events["CustomerID"].notna()
    & events["EventDate"].notna()
    & events["EventType"].notna()
].copy()

stats["Event_Fact"]["missing_required_removed"] = (
    before - len(events)
)

# EventID 主鍵唯一
before = len(events)

events = (
    events
    .sort_values(
        ["EventID", "EventDate"]
    )
    .drop_duplicates(
        subset=["EventID"],
        keep="first"
    )
    .copy()
)

stats["Event_Fact"]["duplicate_pk_removed"] = (
    before - len(events)
)

# Customer FK
before = len(events)

events = events.loc[
    events["CustomerID"].isin(
        valid_customer_ids
    )
].copy()

stats["Event_Fact"]["fk_invalid_removed"] = (
    before - len(events)
)

events["CustomerRegisterDate"] = (
    events["CustomerID"]
    .map(customer_register_map)
)

allowed_event_types = {
    "Register",
    "Login",
    "View",
    "Search",
    "AddCart",
    "Purchase",
    "Review",
    "Referral",
}

before = len(events)

valid_event_mask = (
    date_in_range(events["EventDate"])
    & events["EventDate"].ge(
        events["CustomerRegisterDate"]
    )
    & events["EventType"].isin(
        allowed_event_types
    )
)

events = events.loc[
    valid_event_mask
].copy()

stats["Event_Fact"]["invalid_rows_removed"] = (
    before - len(events)
)

events = events.drop(
    columns=[
        "CustomerRegisterDate"
    ],
    errors="ignore"
)

events = finalize_int_columns(
    events,
    [
        "EventID",
        "CustomerID",
        "SessionID",
    ]
)

# ============================================================
# 8. Referral_Table Cleaning
# ============================================================

print_section("8. CLEANING Referral_Table")

referral = drop_full_duplicates(
    referral,
    stats,
    "Referral_Table"
)

for col in [
    "ReferralID",
    "ReferrerCustomerID",
    "NewCustomerID",
]:
    referral[col] = pd.to_numeric(
        referral[col],
        errors="coerce"
    ).astype("Int64")

referral = convert_datetime(
    referral,
    ["ReferralDate"]
)

before = len(referral)

referral = referral.loc[
    referral["ReferralID"].notna()
    & referral["ReferrerCustomerID"].notna()
    & referral["NewCustomerID"].notna()
    & referral["ReferralDate"].notna()
].copy()

stats["Referral_Table"]["missing_required_removed"] = (
    before - len(referral)
)

# ReferralID 主鍵唯一
before = len(referral)

referral = (
    referral
    .sort_values(
        ["ReferralID", "ReferralDate"]
    )
    .drop_duplicates(
        subset=["ReferralID"],
        keep="first"
    )
    .copy()
)

stats["Referral_Table"]["duplicate_pk_removed"] = (
    before - len(referral)
)

# 外鍵存在
before = len(referral)

referral = referral.loc[
    referral["ReferrerCustomerID"].isin(
        valid_customer_ids
    )
    & referral["NewCustomerID"].isin(
        valid_customer_ids
    )
].copy()

stats["Referral_Table"]["fk_invalid_removed"] = (
    before - len(referral)
)

# 推薦者不能等於被推薦人
# 推薦者 RegisterDate <= ReferralDate <= NewCustomer RegisterDate
referral["ReferrerRegisterDate"] = (
    referral["ReferrerCustomerID"]
    .map(customer_register_map)
)

referral["NewCustomerRegisterDate"] = (
    referral["NewCustomerID"]
    .map(customer_register_map)
)

before = len(referral)

valid_referral_mask = (
    referral["ReferrerCustomerID"].ne(
        referral["NewCustomerID"]
    )
    & date_in_range(
        referral["ReferralDate"]
    )
    & referral["ReferralDate"].ge(
        referral["ReferrerRegisterDate"]
    )
    & referral["ReferralDate"].le(
        referral["NewCustomerRegisterDate"]
    )
)

referral = referral.loc[
    valid_referral_mask
].copy()

stats["Referral_Table"]["invalid_rows_removed"] = (
    before - len(referral)
)

# 一個 NewCustomer 最多一個推薦來源
before = len(referral)

referral = (
    referral
    .sort_values(
        [
            "NewCustomerID",
            "ReferralDate",
            "ReferralID",
        ]
    )
    .drop_duplicates(
        subset=["NewCustomerID"],
        keep="first"
    )
    .copy()
)

stats[
    "Referral_Table"
]["duplicate_new_customer_removed"] = (
    before - len(referral)
)

referral = referral.drop(
    columns=[
        "ReferrerRegisterDate",
        "NewCustomerRegisterDate",
    ],
    errors="ignore"
)

referral = finalize_int_columns(
    referral,
    [
        "ReferralID",
        "ReferrerCustomerID",
        "NewCustomerID",
    ]
)

# ============================================================
# 9. Cross-table Customer Referral Consistency
# ============================================================

print_section("9. CROSS-TABLE CONSISTENCY")

referral_lookup = (
    referral
    .set_index("NewCustomerID")[
        "ReferrerCustomerID"
    ]
    .to_dict()
)

customer["ExpectedReferralUserID"] = (
    customer["CustomerID"]
    .map(referral_lookup)
    .fillna(0)
    .astype("int64")
)

customer_referral_mismatch = (
    customer["ReferralUserID"]
    != customer["ExpectedReferralUserID"]
)

mismatch_count = int(
    customer_referral_mismatch.sum()
)

# 以 Referral_Table 為關係事實來源，
# 將 Customer_Dim.ReferralUserID 校正成一致。
customer.loc[
    customer_referral_mismatch,
    "ReferralUserID"
] = customer.loc[
    customer_referral_mismatch,
    "ExpectedReferralUserID"
]

customer = customer.drop(
    columns=[
        "ExpectedReferralUserID"
    ]
)

stats["Customer_Dim"][
    "referral_relationship_corrected"
] = mismatch_count

# 若 ReferralUserID > 0，AcquisitionSource 應為 Referral
referral_customer_mask = (
    customer["ReferralUserID"] > 0
)

acquisition_corrected = int(
    (
        referral_customer_mask
        & customer[
            "AcquisitionSource"
        ].ne("Referral")
    ).sum()
)

customer.loc[
    referral_customer_mask,
    "AcquisitionSource"
] = "Referral"

stats["Customer_Dim"][
    "acquisition_source_corrected"
] = acquisition_corrected

print(
    "Customer/Referral relationship corrections:",
    f"{mismatch_count:,}"
)

# ============================================================
# 10. Sorting
# ============================================================

print_section("10. SORTING CLEAN DATA")

customer = (
    customer
    .sort_values("CustomerID")
    .reset_index(drop=True)
)

product = (
    product
    .sort_values("ProductID")
    .reset_index(drop=True)
)

campaign = (
    campaign
    .sort_values("CampaignID")
    .reset_index(drop=True)
)

orders = (
    orders
    .sort_values(
        [
            "OrderDate",
            "OrderID",
            "ProductID",
        ]
    )
    .reset_index(drop=True)
)

events = (
    events
    .sort_values(
        [
            "EventDate",
            "EventID",
        ]
    )
    .reset_index(drop=True)
)

referral = (
    referral
    .sort_values("ReferralID")
    .reset_index(drop=True)
)

# ============================================================
# 11. Final Validation
# ============================================================

print_section("11. FINAL ETL VALIDATION")

# ------------------------------------------------------------
# Primary key validation
# ------------------------------------------------------------

validation = {}

validation[
    "CustomerID Unique"
] = customer["CustomerID"].is_unique

validation[
    "ProductID Unique"
] = product["ProductID"].is_unique

validation[
    "CampaignID Unique"
] = campaign["CampaignID"].is_unique

validation[
    "EventID Unique"
] = events["EventID"].is_unique

validation[
    "ReferralID Unique"
] = referral["ReferralID"].is_unique

# OrderID 不要求唯一，因為 Order_Fact 是明細粒度
validation[
    "OrderID Has No Null"
] = orders["OrderID"].notna().all()

# ------------------------------------------------------------
# Foreign keys
# ------------------------------------------------------------

validation[
    "Order Customer FK"
] = orders["CustomerID"].isin(
    set(customer["CustomerID"])
).all()

validation[
    "Order Product FK"
] = orders["ProductID"].isin(
    set(product["ProductID"])
).all()

validation[
    "Order Campaign FK"
] = (
    orders["CampaignID"].eq(0)
    | orders["CampaignID"].isin(
        set(campaign["CampaignID"])
    )
).all()

validation[
    "Event Customer FK"
] = events["CustomerID"].isin(
    set(customer["CustomerID"])
).all()

validation[
    "Referral Referrer FK"
] = referral["ReferrerCustomerID"].isin(
    set(customer["CustomerID"])
).all()

validation[
    "Referral NewCustomer FK"
] = referral["NewCustomerID"].isin(
    set(customer["CustomerID"])
).all()

# ------------------------------------------------------------
# Date validation
# ------------------------------------------------------------

customer_register_check = (
    orders["CustomerID"]
    .map(
        customer.set_index(
            "CustomerID"
        )["RegisterDate"]
    )
)

event_register_check = (
    events["CustomerID"]
    .map(
        customer.set_index(
            "CustomerID"
        )["RegisterDate"]
    )
)

validation[
    "Order >= RegisterDate"
] = (
    orders["OrderDate"]
    >= customer_register_check
).all()

validation[
    "Event >= RegisterDate"
] = (
    events["EventDate"]
    >= event_register_check
).all()

validation[
    "Order Date Range"
] = date_in_range(
    orders["OrderDate"]
).all()

validation[
    "Event Date Range"
] = date_in_range(
    events["EventDate"]
).all()

# ------------------------------------------------------------
# Numeric validation
# ------------------------------------------------------------

validation[
    "Quantity > 0"
] = orders["Quantity"].gt(0).all()

validation[
    "UnitPrice > 0"
] = orders["UnitPrice"].gt(0).all()

validation[
    "GrossAmount > 0"
] = orders["GrossAmount"].gt(0).all()

validation[
    "Amount > 0"
] = orders["Amount"].gt(0).all()

validation[
    "0 <= DiscountRate <= 1"
] = orders["DiscountRate"].between(
    0,
    1,
    inclusive="both"
).all()

validation[
    "DiscountAmount >= 0"
] = orders["DiscountAmount"].ge(0).all()

validation[
    "Product Cost < Price"
] = product["Cost"].lt(
    product["Price"]
).all()

all_validation_passed = all(
    validation.values()
)

for key, result in validation.items():
    print(
        f"{key:<34}:",
        "PASS" if result else "FAIL"
    )

print()
print(
    "Overall Validation:",
    "PASS" if all_validation_passed else "FAIL"
)

if not all_validation_passed:
    failed_items = [
        key
        for key, result
        in validation.items()
        if not result
    ]

    raise ValueError(
        "ETL 最終驗證失敗："
        + ", ".join(failed_items)
    )

# ============================================================
# 12. Export
# ============================================================

print_section("12. EXPORTING ETL OUTPUT")

customer.to_csv(
    OUTPUT_DIR / "Customer_Dim.csv",
    index=False
)

product.to_csv(
    OUTPUT_DIR / "Product_Dim.csv",
    index=False
)

orders.to_csv(
    OUTPUT_DIR / "Order_Fact.csv",
    index=False
)

events.to_csv(
    OUTPUT_DIR / "Event_Fact.csv",
    index=False
)

referral.to_csv(
    OUTPUT_DIR / "Referral_Table.csv",
    index=False
)

campaign.to_csv(
    OUTPUT_DIR / "Marketing_Campaign_Table.csv",
    index=False
)

stats["Customer_Dim"]["output_rows"] = len(customer)
stats["Product_Dim"]["output_rows"] = len(product)
stats["Order_Fact"]["output_rows"] = len(orders)
stats["Event_Fact"]["output_rows"] = len(events)
stats["Referral_Table"]["output_rows"] = len(referral)
stats["Marketing_Campaign_Table"]["output_rows"] = len(campaign)

# ============================================================
# 13. ETL Summary
# ============================================================

print_section("13. ETL SUMMARY")

for table_name, table_stats in stats.items():
    print()
    print(table_name)

    for key, value in table_stats.items():
        if isinstance(value, (int, np.integer)):
            print(
                f"  {key:<38}: {int(value):,}"
            )
        else:
            print(
                f"  {key:<38}: {value}"
            )

print()
print("=" * 86)
print("ETL FINISHED")
print("=" * 86)
print()
print("Output Files:")

for filename in REQUIRED_FILES.values():
    print(
        " -",
        OUTPUT_DIR / filename
    )
