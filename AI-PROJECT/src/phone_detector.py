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

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence=0.35,
        enabled=True,
        person_confidence=None,
        phone_min_aspect_ratio=1.2,
        phone_max_aspect_ratio=4.8,
    ):
        self.enabled = False
        self.available = False
        self.model = None
        self.predict_model = None
        self.confidence = confidence
        self.person_confidence = person_confidence if person_confidence is not None else confidence
        self.phone_min_aspect_ratio = phone_min_aspect_ratio
        self.phone_max_aspect_ratio = phone_max_aspect_ratio
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
            self.predict_model = YOLO(model_path)
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
                    conf=self.person_confidence,
                    verbose=False,
                    persist=True,
                    classes=[0],
                )
                # YOLO tracking can occasionally drop a person even when plain
                # detection still sees them. Keep unmatched detections as
                # no-ID people so the video analyzer can reconnect them.
                person_predict_results = self.predict_model.predict(
                    frame,
                    conf=self.person_confidence,
                    verbose=False,
                    classes=[0],
                )
                phone_results = self.predict_model.predict(
                    frame,
                    conf=self.confidence,
                    verbose=False,
                    classes=[67],
                )
            else:
                results = self.predict_model.predict(frame, conf=self.confidence, verbose=False)
        except Exception as exc:
            self.warning = f"Phone detection failed: {exc}"
            return [], []

        phones = []
        people = []

        if track_people:
            tracked_bboxes = []
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
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    tracked_bboxes.append(bbox)
                    people.append(PersonDetection(bbox, float(box.conf[0]), track_id))

            for result in person_predict_results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, "")).lower()
                    if label != "person":
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    if any(bbox_iou(bbox, tracked_bbox) >= 0.5 for tracked_bbox in tracked_bboxes):
                        continue
                    people.append(PersonDetection(bbox, float(box.conf[0]), None))

            for result in phone_results:
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    label = str(names.get(cls_id, "")).lower()
                    if label not in {"cell phone", "mobile phone", "phone"}:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                    if self._valid_phone_bbox(bbox):
                        phones.append(PhoneDetection(bbox, float(box.conf[0])))
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
                    if self._valid_phone_bbox(bbox):
                        phones.append(PhoneDetection(bbox, confidence))
                elif label == "person":
                    people.append(PersonDetection(bbox, confidence))
        return phones, people

    def _valid_phone_bbox(self, bbox):
        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            return False
        aspect = max(w / h, h / w)
        return self.phone_min_aspect_ratio <= aspect <= self.phone_max_aspect_ratio


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


def bbox_iou(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - inter
    if union <= 0:
        return 0.0
    return inter / union
