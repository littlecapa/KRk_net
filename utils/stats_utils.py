import logging
import json
from utils.param_utils import Params
from utils.file_utils import write_line_to_csv_file, write_summary
from utils.metrics_utils import metrics_to_csv_string

class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
    
class Summary():
    def __init__(self, params):
        self.summary = "Summary:" + "\n"
        self.base_dir = params.base_dir
        self.stats_dir = params.stats_dir

    def add(self, summary, stage = "Training"):
        self.add_stage(stage)
        self.summary += json.dumps(summary) + "\n"

    def add_stage(self, stage):
        self.summary += f"Stage: {stage}"
    
    def get_all(self):
        return self.summary
    
    DEFAULT_SUMMARY_FILE = "summary.txt"
    def save(self):
        write_summary(self.base_dir, self.stats_dir, self.DEFAULT_SUMMARY_FILE, self.get_all())
        self.summary = "Summary:" + "\n"


DEFAULT_STATS_FILE = "stats.csv"   
def save_stats(params, results):
    line = params.get_stats_str()
    line += metrics_to_csv_string(results)
    write_line_to_csv_file(params.base_dir, params.stats_dir, DEFAULT_STATS_FILE, line)