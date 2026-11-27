"""實際故障回歸：舊模型殘留、混版報告、發布中斷回復。"""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import pipeline_runtime as pipeline
from quarterly_models import Predictor


class PipelineSafetyTests(unittest.TestCase):
    def test_constant_export_removes_previous_trees(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            for name in ['profit','classification']:(folder/f'model_{name}.json').write_text('old tree')
            predictor=Predictor('clv','Mean','original').fit(pd.DataFrame({'Actual':[3.,5.]}))
            predictor.export(folder,'model')
            self.assertFalse((folder/'model_profit.json').exists())
            self.assertFalse((folder/'model_classification.json').exists())
            self.assertEqual(json.loads((folder/'model_spec.json').read_text())['ConstantValue'],4.)

    def test_stale_report_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);context={'TargetQuarter':'2026Q1','NamesHash':'old'}
            pipeline.write_json(root/'執行紀錄/已發布批次.json',{'JobID':'123','Context':context})
            pipeline.write_json(root/'報告結果/報告摘要.json',{'JobID':'123','Context':context})
            with patch('segment_contract.context',return_value=context):self.assertTrue(pipeline.published_data_available(root))
            with patch('segment_contract.context',return_value={**context,'NamesHash':'new'}):self.assertFalse(pipeline.published_data_available(root))
            with patch('segment_contract.context',side_effect=ValueError('mixed version')):self.assertFalse(pipeline.published_data_available(root))

    def test_publish_failure_restores_old_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp).resolve();folder=root/'執行紀錄/123';work=folder/'work'
            for parent in [root,work]:
                for name in ['a','b']:
                    (parent/name).mkdir(parents=True,exist_ok=True)
                    (parent/name/'result.txt').write_text('old' if parent==root else 'new')
            replace=pipeline.replace_path
            def fail_once(source,target):
                if Path(source)==work/'b':raise OSError('injected publication failure')
                return replace(source,target)
            with patch.object(pipeline,'publish_targets',return_value=['a','b']),patch.object(pipeline,'replace_path',side_effect=fail_once):
                with self.assertRaisesRegex(OSError,'injected'):pipeline.publish(root,folder,work)
            for name in ['a','b']:self.assertEqual((root/name/'result.txt').read_text(),'old')
            self.assertFalse(pipeline.read_json(folder/'publish.json')['Pending'])


if __name__=='__main__':unittest.main()
