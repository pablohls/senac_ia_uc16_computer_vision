"""Wrapper fino sobre a YOLO (ultralytics).

Mantem as ultimas deteccoes em cache para que, nos frames em que o gatilho NAO
aciona a inferencia, possamos reaproveitar as caixas do ultimo frame detectado.
"""

import numpy as np
from ultralytics import YOLO


class Detector:
    """Carrega a YOLO uma unica vez e roda inferencia sob demanda."""

    def __init__(self, model_path: str = "yolov8m.pt") -> None:
        # O peso e baixado automaticamente pela ultralytics no primeiro uso.
        self.model = YOLO(model_path)
        self._last_annotated: np.ndarray | None = None
        self._last_count: int = 0

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, int]:
        """Roda a YOLO no frame e retorna (frame_anotado, numero_de_deteccoes).

        verbose=False evita poluir o terminal com o log de cada inferencia.
        """
        results = self.model(frame, verbose=False)
        result = results[0]
        annotated = result.plot()  # desenha caixas/labels sobre uma copia do frame
        count = len(result.boxes)

        self._last_annotated = annotated
        self._last_count = count
        return annotated, count

    def last_annotated(self, fallback: np.ndarray) -> np.ndarray:
        """Retorna a ultima anotacao para reuso em frames pulados.

        Se ainda nao houve nenhuma deteccao, devolve o frame original (fallback).
        """
        return self._last_annotated if self._last_annotated is not None else fallback

    @property
    def last_count(self) -> int:
        return self._last_count
