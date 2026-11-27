"""由目標季度推導固定12季歷史窗；每個批次保存自己的設定。"""
from dataclasses import dataclass
from pathlib import Path
import json
import re
import pandas as pd

@dataclass(frozen=True)
class QuarterPlan:
    target: str = '2026Q1'

    def __post_init__(self):
        if not re.fullmatch(r'20\d{2}Q[1-4]',self.target):
            raise ValueError('目標季度格式須為 YYYYQ1～YYYYQ4，例如2026Q2。')

    @property
    def forecast(self):return self.target
    @property
    def history_start(self):return str(pd.Period(self.target,freq='Q')-12)
    @property
    def history_end(self):return str(pd.Period(self.target,freq='Q')-2)
    @property
    def holdout(self):return str(pd.Period(self.target,freq='Q')-1)
    @property
    def first_label(self):return str(pd.Period(self.target,freq='Q')-11)
    @property
    def start_date(self):return str(pd.Period(self.history_start).start_time.date())
    @property
    def development_end(self):return str(pd.Period(self.history_end).end_time.date())
    @property
    def score_end(self):return str(pd.Period(self.holdout).end_time.date())
    @property
    def forecast_end(self):return str(pd.Period(self.target).end_time.date())
    @property
    def quarters(self):return list(map(str,pd.period_range(self.history_start,self.holdout,freq='Q')))

def load_plan(root):
    path=Path(root)/'季度設定.json'
    if not path.exists():return QuarterPlan()
    return QuarterPlan(json.loads(path.read_text(encoding='utf-8'))['TargetQuarter'])
