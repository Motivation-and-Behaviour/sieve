import json
import os
from datetime import datetime


class BatchTracker:
    def __init__(self, filepath="batches.json"):
        self.filepath = filepath
        if not os.path.exists(filepath):
            self._save({})

    def _load(self):
        with open(self.filepath) as f:
            return json.load(f)

    def _save(self, data):
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)

    def add_batch(self, batch_id, batch_type):
        data = self._load()
        data[batch_id] = {
            "type": batch_type,  # 'abstract', 'fulltext', or 'extraction'
            "status": "in_progress",
            "created_at": datetime.now().isoformat(),
        }
        self._save(data)

    def get_pending_batches(self):
        data = self._load()
        return {k: v for k, v in data.items() if v["status"] != "completed"}

    def mark_completed(self, batch_id):
        data = self._load()
        if batch_id in data:
            data[batch_id]["status"] = "completed"
            self._save(data)
