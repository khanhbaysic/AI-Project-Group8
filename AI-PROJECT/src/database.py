import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StudentRecord:
    student_id: str
    name: str
    reference_image: Path


class StudentDatabase:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self.records: dict[str, StudentRecord] = {}
        self.load()

    def load(self):
        self.records.clear()
        if not self.csv_path.exists():
            return

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                student_id = (row.get("student_id") or "").strip()
                if not student_id:
                    continue
                name = (row.get("name") or "").strip()
                image_value = (row.get("reference_image") or "").strip()
                image_path = Path(image_value)
                if not image_path.is_absolute():
                    image_path = self.csv_path.parent / image_path
                self.records[student_id] = StudentRecord(student_id, name, image_path)

    def get(self, student_id: str):
        return self.records.get(student_id.strip())
