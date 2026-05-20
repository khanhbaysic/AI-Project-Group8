from dataclasses import dataclass


@dataclass
class PhoneDetection:
    bbox: tuple[int, int, int, int]
    confidence: float


@dataclass
class PersonDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    track_id: int | None = None


class PhoneDetector:
    """Optional YOLO-based cell phone detector.

    The project keeps this module isolated because object detection needs an
    extra dependency (`ultralytics`) and model weights. If the dependency is not
    installed, the detector disables itself safely and the rest of the system
    still runs.
    """

    def __init__(self, model_path="yolov8n.pt", confidence=0.35, enabled=True):
        self.enabled = False
        self.available = False
        self.model = None
        self.confidence = confidence
        self.warning = ""

        if not enabled:
            self.warning = "Phone detection disabled by config."
            return

        try:
            from ultralytics import YOLO
        except Exception:
            self.warning = "ultralytics is not installed. Phone detection disabled."
            return

        try:
            self.model = YOLO(model_path)
        except Exception as exc:
            self.warning = f"Could not load phone detection model '{model_path}': {exc}"
            return

        self.enabled = True
        self.available = True

    def detect(self, frame):
        phones, _ = self.detect_people_and_phones(frame)
        return phones

    def detect_people_and_phones(self, frame, track_people=False):
        if not self.available:
            return [], []

        try:
            if track_people:
                person_results = self.model.track(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    persist=True,
                    classes=[0],
                )
                phone_results = self.model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    classes=[67],
                )
            else:
                results = self.model.predict(frame, conf=self.confidence, verbose=False)
        except Exception as exc:
            self.warning = f"Phone detection failed: {exc}"
            return [], []

        phones = []
        people = []

        if track_people:
            for result in person_results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, "")).lower()
                    if label != "person":
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    track_id = None
                    if getattr(box, "id", None) is not None:
                        track_id = int(box.id[0])
                    people.append(PersonDetection((x1, y1, x2 - x1, y2 - y1), float(box.conf[0]), track_id))

            for result in phone_results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, "")).lower()
                    if label not in {"cell phone", "mobile phone", "phone"}:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    phones.append(PhoneDetection((x1, y1, x2 - x1, y2 - y1), float(box.conf[0])))
            return phones, people

        for result in results:
            names = result.names
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = str(names.get(cls_id, "")).lower()
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                confidence = float(box.conf[0])
                bbox = (x1, y1, x2 - x1, y2 - y1)
                if label in {"cell phone", "mobile phone", "phone"}:
                    phones.append(PhoneDetection(bbox, confidence))
                elif label == "person":
                    people.append(PersonDetection(bbox, confidence))
        return phones, people


def expanded_intersection(bbox_a, bbox_b, scale=1.8):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    cx = ax + aw / 2.0
    cy = ay + ah / 2.0
    ew = aw * scale
    eh = ah * scale
    ex = cx - ew / 2.0
    ey = cy - eh / 2.0

    ix1 = max(ex, bx)
    iy1 = max(ey, by)
    ix2 = min(ex + ew, bx + bw)
    iy2 = min(ey + eh, by + bh)
    return ix2 > ix1 and iy2 > iy1
