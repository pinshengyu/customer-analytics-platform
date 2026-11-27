import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Data Warehouse V1
# ETL_Output -> Star Schema
#
# 本程式只負責：
# - 建立維度表
# - 建立事實表
# - 建立 Surrogate Key
# - 建立維度 / 事實表關聯
# - 驗證 Data Warehouse 關聯完整性
#
# 本程式不負責：
# - RAW Data 清理
# - ETL
# - RFM / Cohort / AARRR
# - K-Means
# - ChurnLabel
# - Future_12M_Revenue
# - Churn / CLV Prediction
#
# 輸入：
#   ETL_Output/
#
# 輸出：
#   原始資料/Data_Warehouse/
#
# DW Tables：
#   Dim_Customer.csv
#   Dim_Product.csv
#   Dim_Time.csv
#   Dim_Campaign.csv
#   Dim_Store.csv
#   Fact_Sales.csv
#   Fact_Event.csv
#   Fact_Referral.csv
#
# Store 相容設計：
#   若未來 ETL_Output/Store_Dim.csv 存在，
#   會自動讀取完整 Store 維度。
#
#   若目前尚未有 Store_Dim.csv，
#   則由 Order_Fact.StoreID 建立最基本的 Dim_Store。
# ============================================================



# ============================================================
# 1. Project Root Discovery
# ============================================================

def find_project_root():
    """
    從本程式所在位置一路往上尋找具有 ETL_Output 的專題根目錄。
    不使用固定 parent 層數。
    """
    current = Path(__file__).resolve().parent

    for parent in [current, *current.parents]:
        if (parent / "ETL_Output").is_dir():
            return parent

    raise FileNotFoundError(
        "找不到專題根目錄：未找到 ETL_Output 資料夾。"
    )


PROJECT_DIR = find_project_root()
import sys
sys.path.insert(0,str(PROJECT_DIR))
from quarterly_plan import load_plan
PLAN = load_plan(PROJECT_DIR)
START_DATE = pd.Timestamp(PLAN.start_date)
END_DATE = pd.Timestamp(PLAN.score_end)
ETL_DIR = PROJECT_DIR / "ETL_Output"
DW_DIR = PROJECT_DIR / "原始資料" / "Data_Warehouse"

DW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("DATA WAREHOUSE V1 - STAR SCHEMA BUILDER")
print("=" * 90)
print("Project Root :", PROJECT_DIR)
print("ETL Input    :", ETL_DIR)
print("DW Output    :", DW_DIR)

# ============================================================
# 2. Required Files
# ============================================================

REQUIRED_FILES = {
    "customer": "Customer_Dim.csv",
    "product": "Product_Dim.csv",
    "order": "Order_Fact.csv",
    "event": "Event_Fact.csv",
    "referral": "Referral_Table.csv",
    "campaign": "Marketing_Campaign_Table.csv",
}

for name, filename in REQUIRED_FILES.items():
    path = ETL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 ETL 輸入檔案：{path}"
        )


def print_section(title):
    print()
    print("-" * 90)
    print(title)
    print("-" * 90)


def load_csv(path, name):
    df = pd.read_csv(path, low_memory=False)
    print(f"{name:<32} rows = {len(df):,}")
    return df


def require_columns(df, table_name, columns):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"{table_name} 缺少必要欄位：{missing}"
        )


def to_datetime_columns(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(
                df[col],
                errors="coerce"
            )
    return df


# ============================================================
# 3. Load ETL Output
# ============================================================

print_section("1. LOADING ETL OUTPUT")

customer = load_csv(
    ETL_DIR / REQUIRED_FILES["customer"],
    "Customer_Dim"
)

product = load_csv(
    ETL_DIR / REQUIRED_FILES["product"],
    "Product_Dim"
)

orders = load_csv(
    ETL_DIR / REQUIRED_FILES["order"],
    "Order_Fact"
)

events = load_csv(
    ETL_DIR / REQUIRED_FILES["event"],
    "Event_Fact"
)

referral = load_csv(
    ETL_DIR / REQUIRED_FILES["referral"],
    "Referral_Table"
)

campaign = load_csv(
    ETL_DIR / REQUIRED_FILES["campaign"],
    "Marketing_Campaign_Table"
)

# 日期型別
customer = to_datetime_columns(
    customer,
    ["RegisterDate"]
)

product = to_datetime_columns(
    product,
    ["LaunchDate"]
)

orders = to_datetime_columns(
    orders,
    ["OrderDate"]
)

events = to_datetime_columns(
    events,
    ["EventDate"]
)

referral = to_datetime_columns(
    referral,
    ["ReferralDate"]
)

campaign = to_datetime_columns(
    campaign,
    ["StartDate", "EndDate"]
)

# ============================================================
# 4. Input Schema Validation
# ============================================================

print_section("2. INPUT SCHEMA VALIDATION")

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

print("Input schema validation: PASS")

# ============================================================
# 5. Build Dim_Customer
# ============================================================

print_section("3. BUILDING Dim_Customer")

dim_customer = (
    customer
    .sort_values("CustomerID")
    .reset_index(drop=True)
    .copy()
)

dim_customer.insert(
    0,
    "CustomerKey",
    np.arange(
        1,
        len(dim_customer) + 1,
        dtype=np.int64
    )
)

# 若有 ReferralUserID，補上其對應的 Surrogate Key。
customer_key_map = (
    dim_customer
    .set_index("CustomerID")["CustomerKey"]
    .to_dict()
)

dim_customer["ReferralCustomerKey"] = (
    dim_customer["ReferralUserID"]
    .map(customer_key_map)
    .fillna(0)
    .astype("int64")
)

# 沒有推薦人的 Customer 使用 0
dim_customer.loc[
    dim_customer["ReferralUserID"].eq(0),
    "ReferralCustomerKey"
] = 0

# ============================================================
# 6. Build Dim_Product
# ============================================================

print_section("4. BUILDING Dim_Product")

dim_product = (
    product
    .sort_values("ProductID")
    .reset_index(drop=True)
    .copy()
)

dim_product.insert(
    0,
    "ProductKey",
    np.arange(
        1,
        len(dim_product) + 1,
        dtype=np.int64
    )
)

product_key_map = (
    dim_product
    .set_index("ProductID")["ProductKey"]
    .to_dict()
)

# 商品成本對照：供 Fact_Sales 建立實際成本與利潤欄位。
product_cost_map = (
    dim_product
    .set_index("ProductID")["Cost"]
    .to_dict()
)

# ============================================================
# 7. Build Dim_Time
# ============================================================

print_section("5. BUILDING Dim_Time")

date_range = pd.date_range(
    START_DATE,
    END_DATE,
    freq="D"
)

dim_time = pd.DataFrame({
    "Date": date_range
})

dim_time["DateKey"] = (
    dim_time["Date"]
    .dt.strftime("%Y%m%d")
    .astype("int64")
)

dim_time["Year"] = dim_time["Date"].dt.year
dim_time["Quarter"] = dim_time["Date"].dt.quarter
dim_time["Month"] = dim_time["Date"].dt.month
dim_time["MonthName"] = dim_time["Date"].dt.month_name()
dim_time["YearMonth"] = (
    dim_time["Date"]
    .dt.to_period("M")
    .astype(str)
)
dim_time["Day"] = dim_time["Date"].dt.day
dim_time["DayOfWeek"] = dim_time["Date"].dt.dayofweek + 1
dim_time["DayName"] = dim_time["Date"].dt.day_name()
dim_time["WeekOfYear"] = (
    dim_time["Date"]
    .dt.isocalendar()
    .week
    .astype("int64")
)
dim_time["IsWeekend"] = (
    dim_time["Date"]
    .dt.dayofweek
    .isin([5, 6])
    .astype("int64")
)

# DateKey 放最前面
dim_time = dim_time[
    [
        "DateKey",
        "Date",
        "Year",
        "Quarter",
        "Month",
        "MonthName",
        "YearMonth",
        "Day",
        "DayOfWeek",
        "DayName",
        "WeekOfYear",
        "IsWeekend",
    ]
]

date_key_map = (
    dim_time
    .set_index("Date")["DateKey"]
    .to_dict()
)

# ============================================================
# 8. Build Dim_Campaign
# ============================================================

print_section("6. BUILDING Dim_Campaign")

dim_campaign_real = (
    campaign
    .sort_values("CampaignID")
    .reset_index(drop=True)
    .copy()
)

dim_campaign_real.insert(
    0,
    "CampaignKey",
    np.arange(
        1,
        len(dim_campaign_real) + 1,
        dtype=np.int64
    )
)

# CampaignID = 0 代表沒有活動。
# 在 DW 中建立正式的 No Campaign 維度成員。
no_campaign_row = pd.DataFrame({
    "CampaignKey": [0],
    "CampaignID": [0],
    "CampaignName": ["No Campaign"],
    "Channel": ["None"],
    "Budget": [0],
    "StartDate": [pd.NaT],
    "EndDate": [pd.NaT],
    "DiscountRate": [0.0],
    "CampaignType": ["None"],
})

# 若 Campaign 未來增加額外欄位，
# 自動補 NA，避免 concat 欄位遺失。
for col in dim_campaign_real.columns:
    if col not in no_campaign_row.columns:
        no_campaign_row[col] = pd.NA

for col in no_campaign_row.columns:
    if col not in dim_campaign_real.columns:
        dim_campaign_real[col] = pd.NA

dim_campaign = pd.concat(
    [
        no_campaign_row[
            dim_campaign_real.columns
        ],
        dim_campaign_real
    ],
    ignore_index=True
)

campaign_key_map = (
    dim_campaign
    .set_index("CampaignID")["CampaignKey"]
    .to_dict()
)

# ============================================================
# 9. Build Dim_Store
# ============================================================

print_section("7. BUILDING Dim_Store")

store_source_path = ETL_DIR / "Store_Dim.csv"

if store_source_path.exists():
    print(
        "Store source detected:",
        store_source_path
    )

    store_source = pd.read_csv(
        store_source_path,
        low_memory=False
    )

    require_columns(
        store_source,
        "Store_Dim",
        ["StoreID"]
    )

    if "OpenDate" in store_source.columns:
        store_source["OpenDate"] = pd.to_datetime(
            store_source["OpenDate"],
            errors="coerce"
        )

    store_source = (
        store_source
        .drop_duplicates(
            subset=["StoreID"],
            keep="first"
        )
        .sort_values("StoreID")
        .reset_index(drop=True)
    )

    # 僅保留真的有出現在銷售資料中的門市，
    # 同時若有 ETL Store 表中額外但尚無交易的門市，也保留。
    dim_store = store_source.copy()

else:
    print(
        "Store_Dim.csv not found."
    )
    print(
        "Building basic Dim_Store from Order_Fact.StoreID."
    )

    dim_store = pd.DataFrame({
        "StoreID": sorted(
            orders["StoreID"]
            .dropna()
            .astype("int64")
            .unique()
        )
    })

dim_store.insert(
    0,
    "StoreKey",
    np.arange(
        1,
        len(dim_store) + 1,
        dtype=np.int64
    )
)

store_key_map = (
    dim_store
    .set_index("StoreID")["StoreKey"]
    .to_dict()
)

# ============================================================
# 10. Build Fact_Sales
# ============================================================

print_section("8. BUILDING Fact_Sales")

fact_sales = orders.copy()

fact_sales["CustomerKey"] = (
    fact_sales["CustomerID"]
    .map(customer_key_map)
)

fact_sales["ProductKey"] = (
    fact_sales["ProductID"]
    .map(product_key_map)
)

fact_sales["DateKey"] = (
    fact_sales["OrderDate"]
    .dt.normalize()
    .map(date_key_map)
)

fact_sales["CampaignKey"] = (
    fact_sales["CampaignID"]
    .map(campaign_key_map)
)

fact_sales["StoreKey"] = (
    fact_sales["StoreID"]
    .map(store_key_map)
)

# UnitCost 來自 Dim_Product.Cost。
# 每列為 Order Line，因此 TotalCost = Quantity * UnitCost。
# Profit = 折扣後實際營收 Amount - TotalCost。
fact_sales["UnitCost"] = (
    fact_sales["ProductID"]
    .map(product_cost_map)
)

fact_sales["TotalCost"] = (
    fact_sales["Quantity"] * fact_sales["UnitCost"]
)

fact_sales["Profit"] = (
    fact_sales["Amount"] - fact_sales["TotalCost"]
)

fact_sales["ProfitMargin"] = np.where(
    fact_sales["Amount"].ne(0),
    fact_sales["Profit"] / fact_sales["Amount"],
    np.nan
)

# 每一列為一筆 Order Line，因此建立 SalesKey。
fact_sales.insert(
    0,
    "SalesKey",
    np.arange(
        1,
        len(fact_sales) + 1,
        dtype=np.int64
    )
)

# Fact_Sales 僅保留 DW 所需欄位。
# OrderID 為 Degenerate Dimension。
fact_sales = fact_sales[
    [
        "SalesKey",
        "OrderID",
        "CustomerKey",
        "ProductKey",
        "DateKey",
        "CampaignKey",
        "StoreKey",
        "Channel",
        "Quantity",
        "UnitPrice",
        "GrossAmount",
        "DiscountRate",
        "DiscountAmount",
        "Amount",
        "UnitCost",
        "TotalCost",
        "Profit",
        "ProfitMargin",
    ]
].copy()

for col in [
    "CustomerKey",
    "ProductKey",
    "DateKey",
    "CampaignKey",
    "StoreKey",
]:
    fact_sales[col] = (
        fact_sales[col]
        .astype("int64")
    )

# ============================================================
# 11. Build Fact_Event
# ============================================================

print_section("9. BUILDING Fact_Event")

fact_event = events.copy()

fact_event["CustomerKey"] = (
    fact_event["CustomerID"]
    .map(customer_key_map)
)

fact_event["DateKey"] = (
    fact_event["EventDate"]
    .dt.normalize()
    .map(date_key_map)
)

fact_event.insert(
    0,
    "EventFactKey",
    np.arange(
        1,
        len(fact_event) + 1,
        dtype=np.int64
    )
)

fact_event = fact_event[
    [
        "EventFactKey",
        "EventID",
        "CustomerKey",
        "DateKey",
        "EventDate",
        "EventType",
        "Channel",
        "SessionID",
    ]
].copy()

fact_event["CustomerKey"] = (
    fact_event["CustomerKey"]
    .astype("int64")
)

fact_event["DateKey"] = (
    fact_event["DateKey"]
    .astype("int64")
)

# ============================================================
# 12. Build Fact_Referral
# ============================================================

print_section("10. BUILDING Fact_Referral")

fact_referral = referral.copy()

fact_referral["ReferrerCustomerKey"] = (
    fact_referral["ReferrerCustomerID"]
    .map(customer_key_map)
)

fact_referral["NewCustomerKey"] = (
    fact_referral["NewCustomerID"]
    .map(customer_key_map)
)

fact_referral["DateKey"] = (
    fact_referral["ReferralDate"]
    .dt.normalize()
    .map(date_key_map)
)

fact_referral.insert(
    0,
    "ReferralFactKey",
    np.arange(
        1,
        len(fact_referral) + 1,
        dtype=np.int64
    )
)

fact_referral = fact_referral[
    [
        "ReferralFactKey",
        "ReferralID",
        "ReferrerCustomerKey",
        "NewCustomerKey",
        "DateKey",
        "ReferralDate",
    ]
].copy()

for col in [
    "ReferrerCustomerKey",
    "NewCustomerKey",
    "DateKey",
]:
    fact_referral[col] = (
        fact_referral[col]
        .astype("int64")
    )

# ============================================================
# 13. DW Validation
# ============================================================

print_section("11. DATA WAREHOUSE VALIDATION")

validation = {}

# Dimension key uniqueness
validation["Dim_Customer CustomerKey Unique"] = (
    dim_customer["CustomerKey"].is_unique
)
validation["Dim_Customer CustomerID Unique"] = (
    dim_customer["CustomerID"].is_unique
)

validation["Dim_Product ProductKey Unique"] = (
    dim_product["ProductKey"].is_unique
)
validation["Dim_Product ProductID Unique"] = (
    dim_product["ProductID"].is_unique
)

validation["Dim_Time DateKey Unique"] = (
    dim_time["DateKey"].is_unique
)
validation["Dim_Time Date Unique"] = (
    dim_time["Date"].is_unique
)

validation["Dim_Campaign CampaignKey Unique"] = (
    dim_campaign["CampaignKey"].is_unique
)
validation["Dim_Campaign CampaignID Unique"] = (
    dim_campaign["CampaignID"].is_unique
)

validation["Dim_Store StoreKey Unique"] = (
    dim_store["StoreKey"].is_unique
)
validation["Dim_Store StoreID Unique"] = (
    dim_store["StoreID"].is_unique
)

# Fact surrogate keys
validation["Fact_Sales SalesKey Unique"] = (
    fact_sales["SalesKey"].is_unique
)
validation["Fact_Event EventFactKey Unique"] = (
    fact_event["EventFactKey"].is_unique
)
validation["Fact_Referral ReferralFactKey Unique"] = (
    fact_referral["ReferralFactKey"].is_unique
)

# Row count preservation
validation["Fact_Sales Row Count Preserved"] = (
    len(fact_sales) == len(orders)
)
validation["Fact_Event Row Count Preserved"] = (
    len(fact_event) == len(events)
)
validation["Fact_Referral Row Count Preserved"] = (
    len(fact_referral) == len(referral)
)

# Cost / Profit validation
validation["Fact_Sales UnitCost No Null"] = (
    fact_sales["UnitCost"].notna().all()
)
validation["Fact_Sales TotalCost Valid"] = np.allclose(
    fact_sales["TotalCost"],
    fact_sales["Quantity"] * fact_sales["UnitCost"],
    rtol=1e-9,
    atol=1e-6,
    equal_nan=False
)
validation["Fact_Sales Profit Valid"] = np.allclose(
    fact_sales["Profit"],
    fact_sales["Amount"] - fact_sales["TotalCost"],
    rtol=1e-9,
    atol=1e-6,
    equal_nan=False
)

# Fact FK validation
customer_keys = set(
    dim_customer["CustomerKey"]
)
product_keys = set(
    dim_product["ProductKey"]
)
date_keys = set(
    dim_time["DateKey"]
)
campaign_keys = set(
    dim_campaign["CampaignKey"]
)
store_keys = set(
    dim_store["StoreKey"]
)

validation["Fact_Sales Customer FK"] = (
    fact_sales["CustomerKey"]
    .isin(customer_keys)
    .all()
)
validation["Fact_Sales Product FK"] = (
    fact_sales["ProductKey"]
    .isin(product_keys)
    .all()
)
validation["Fact_Sales Date FK"] = (
    fact_sales["DateKey"]
    .isin(date_keys)
    .all()
)
validation["Fact_Sales Campaign FK"] = (
    fact_sales["CampaignKey"]
    .isin(campaign_keys)
    .all()
)
validation["Fact_Sales Store FK"] = (
    fact_sales["StoreKey"]
    .isin(store_keys)
    .all()
)

validation["Fact_Event Customer FK"] = (
    fact_event["CustomerKey"]
    .isin(customer_keys)
    .all()
)
validation["Fact_Event Date FK"] = (
    fact_event["DateKey"]
    .isin(date_keys)
    .all()
)

validation["Fact_Referral Referrer FK"] = (
    fact_referral["ReferrerCustomerKey"]
    .isin(customer_keys)
    .all()
)
validation["Fact_Referral NewCustomer FK"] = (
    fact_referral["NewCustomerKey"]
    .isin(customer_keys)
    .all()
)
validation["Fact_Referral Date FK"] = (
    fact_referral["DateKey"]
    .isin(date_keys)
    .all()
)

# Null keys
validation["Fact_Sales No Null DW Keys"] = (
    fact_sales[
        [
            "CustomerKey",
            "ProductKey",
            "DateKey",
            "CampaignKey",
            "StoreKey",
        ]
    ]
    .notna()
    .all()
    .all()
)

validation["Fact_Event No Null DW Keys"] = (
    fact_event[
        [
            "CustomerKey",
            "DateKey",
        ]
    ]
    .notna()
    .all()
    .all()
)

validation["Fact_Referral No Null DW Keys"] = (
    fact_referral[
        [
            "ReferrerCustomerKey",
            "NewCustomerKey",
            "DateKey",
        ]
    ]
    .notna()
    .all()
    .all()
)

all_validation_passed = all(
    validation.values()
)

for item, result in validation.items():
    print(
        f"{item:<44}:",
        "PASS" if result else "FAIL"
    )

print()
print(
    "Overall DW Validation:",
    "PASS" if all_validation_passed else "FAIL"
)

if not all_validation_passed:
    failed_items = [
        item
        for item, result in validation.items()
        if not result
    ]
    raise ValueError(
        "Data Warehouse 驗證失敗："
        + ", ".join(failed_items)
    )

# ============================================================
# 14. Export
# ============================================================

print_section("12. EXPORTING DATA WAREHOUSE")

output_tables = {
    "Dim_Customer.csv": dim_customer,
    "Dim_Product.csv": dim_product,
    "Dim_Time.csv": dim_time,
    "Dim_Campaign.csv": dim_campaign,
    "Dim_Store.csv": dim_store,
    "Fact_Sales.csv": fact_sales,
    "Fact_Event.csv": fact_event,
    "Fact_Referral.csv": fact_referral,
}

for filename, df in output_tables.items():
    output_path = DW_DIR / filename
    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"{filename:<26} rows = {len(df):,}"
    )

# ============================================================
# 15. Summary
# ============================================================

print_section("13. DATA WAREHOUSE SUMMARY")

print("Dimensions")
print(
    f"  Dim_Customer : {len(dim_customer):,}"
)
print(
    f"  Dim_Product  : {len(dim_product):,}"
)
print(
    f"  Dim_Time     : {len(dim_time):,}"
)
print(
    f"  Dim_Campaign : {len(dim_campaign):,}"
)
print(
    f"  Dim_Store    : {len(dim_store):,}"
)

print()
print("Facts")
print(
    f"  Fact_Sales    : {len(fact_sales):,}"
)
print(
    f"  Fact_Event    : {len(fact_event):,}"
)
print(
    f"  Fact_Referral : {len(fact_referral):,}"
)

print()
print("Fact_Sales Grain:")
print(
    "  One row = one order product line"
)
print(
    "  OrderID may repeat across multiple ProductKey values."
)

print()
print("Fact_Event Grain:")
print(
    "  One row = one customer event."
)

print()
print("Fact_Referral Grain:")
print(
    "  One row = one referral relationship."
)

print()
print("Special Dimension Members:")
print(
    "  CampaignKey = 0 -> No Campaign"
)

print()
if store_source_path.exists():
    print(
        "Dim_Store Source: ETL_Output/Store_Dim.csv"
    )
else:
    print(
        "Dim_Store Source: Distinct StoreID from Order_Fact"
    )

print()
print("=" * 90)
print("DATA WAREHOUSE BUILD FINISHED")
print("=" * 90)

print()
print("Output Files:")
for filename in output_tables:
    print(" -", DW_DIR / filename)
