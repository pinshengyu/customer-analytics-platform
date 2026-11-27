import unittest
import pandas as pd
import quarterly_data as qd
from quarterly_plan import QuarterPlan
from pathlib import Path
import json
import tempfile
from unittest.mock import patch


class QuarterlyTests(unittest.TestCase):
    def test_rolling_plan_year_boundaries(self):
        for target,start,end,holdout in [('2026Q1','2023Q1','2025Q3','2025Q4'),('2026Q2','2023Q2','2025Q4','2026Q1'),('2027Q1','2024Q1','2026Q3','2026Q4')]:
            p=QuarterPlan(target)
            self.assertEqual((p.history_start,p.history_end,p.holdout),(start,end,holdout))
            self.assertEqual(len(p.quarters),12)
        self.assertEqual(QuarterPlan('2024Q2').score_end,'2024-03-31')
        with self.assertRaises(ValueError):QuarterPlan('2026Q5')

    def test_later_target_builds_shifted_labels(self):
        plan=QuarterPlan('2027Q3')
        rows=[{'CustomerID':str(i),'OrderID':f'{q}-{i}','Date':q.start_time,'Revenue':100.,'Profit':20.}
              for q in pd.period_range(plan.history_start,plan.holdout,freq='Q') for i in range(20)]
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'季度設定.json').write_text(json.dumps({'TargetQuarter':plan.target}),encoding='utf-8')
            with patch.object(qd,'load_sales',return_value=pd.DataFrame(rows)):
                _,history,holdout,forecast=qd.build(root)
            self.assertEqual(history.TargetQuarter.min(),plan.first_label)
            self.assertEqual(history.TargetQuarter.max(),plan.history_end)
            self.assertTrue(holdout.TargetQuarter.eq('2027Q2').all())
            self.assertTrue(forecast.TargetQuarter.eq('2027Q3').all())
            self.assertTrue(forecast.ObservationStart.eq('2026-07-01').all())
            self.assertFalse(any(c.startswith('Actual') for c in forecast))

    def test_input_requires_selected_twelve_quarters(self):
        from pipeline_runtime import SCHEMAS,DATE_COLUMNS,validate_inputs
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name,columns in SCHEMAS.items():
                data=pd.DataFrame([{c:'x' for c in columns} for _ in range(12)])
                for col in DATE_COLUMNS[name]:data[col]=[str(q.start_time.date()) for q in pd.period_range('2023Q1','2025Q4',freq='Q')]
                if name=='Customer_Dim.csv':data['RegisterDate']='2020-01-01'
                data.to_csv(root/name,index=False)
            self.assertEqual(len(validate_inputs(root,'2026Q1')),6)
            with self.assertRaisesRegex(ValueError,'2026Q1'):validate_inputs(root,'2026Q2')

    def sales(self):
        return pd.DataFrame({'CustomerID':['A','A','A','B','B'], 'OrderID':[1,2,3,4,5],
            'Date':pd.to_datetime(['2024-12-31','2025-03-31','2025-04-01','2024-12-30','2025-01-01']),
            'Revenue':[100,20,30,40,50],'Profit':[10,-5,6,4,0]})

    def test_no_future_features_and_quarter_labels(self):
        s=self.sales();before=qd.snapshot(s,'2025Q1').set_index('CustomerID')
        changed=s.copy();changed.loc[changed.Date.ge('2025-01-01'),['Revenue','Profit']]=99999
        after=qd.snapshot(changed,'2025Q1').set_index('CustomerID')
        pd.testing.assert_frame_equal(before[qd.ENHANCED],after[qd.ENHANCED])
        self.assertEqual(before.loc['A','ActualProfit'],-5)
        self.assertEqual(before.loc['B','ActualProfit'],0)
        self.assertEqual(before.loc['B','Buyer'],1) # 零利潤仍然是購買
        self.assertEqual(before.loc['A','Recency'],0)

    def test_forecast_has_no_label_and_rolling_window(self):
        snapshot=qd.snapshot(self.sales(),'2026Q1')
        self.assertFalse(any(c.startswith('Actual') for c in snapshot))
        self.assertTrue(snapshot.ObservationStart.eq('2025-01-01').all())
        self.assertEqual(snapshot.set_index('CustomerID').loc['A','Frequency'],2)

    def test_partial_quarter_split_purges_labels(self):
        records=[]
        for q in pd.period_range('2023Q2','2025Q3',freq='Q'):
            for i in range(11):records.append({'SampleID':f'{i}@{q}','CustomerID':str(i),'TargetQuarter':str(q),'ObservationEnd':str((q.start_time-pd.Timedelta(days=1)).date()),'LabelEnd':str(q.end_time.date())})
        d=qd.split_samples(pd.DataFrame(records));used=d.loc[d.Eligible]
        self.assertEqual(d.Split.eq('Train').sum(),77)
        self.assertTrue((~d.Eligible).any())
        self.assertLessEqual(used.loc[used.Split.eq('Train'),'LabelEnd'].max(),used.loc[used.Split.eq('Validation'),'ObservationEnd'].min())
        self.assertLessEqual(used.loc[used.Split.eq('Validation'),'LabelEnd'].max(),used.loc[used.Split.eq('Test'),'ObservationEnd'].min())
        self.assertTrue(d.SampleID.is_unique)


if __name__=='__main__':unittest.main()
