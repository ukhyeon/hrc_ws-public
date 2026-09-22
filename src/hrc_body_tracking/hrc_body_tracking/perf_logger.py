import csv
import os
import threading


class PerfCsvLogger:
    def __init__(self, path, fieldnames, flush_every=30):
        self.path = path
        self.fieldnames = list(fieldnames)
        self.flush_every = max(1, int(flush_every))
        self._count = 0
        self._lock = threading.Lock()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._file = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def write(self, **values):
        row = {name: values.get(name, "") for name in self.fieldnames}
        with self._lock:
            self._writer.writerow(row)
            self._count += 1
            if self._count % self.flush_every == 0:
                self._file.flush()

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.flush()
                self._file.close()
