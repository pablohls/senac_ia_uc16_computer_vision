"""Gatilho de inferencia baseado em diferenca entre frames (CV classica).

A ideia central do mini-projeto: se o frame atual e suficientemente parecido com
o anterior, nao ha nova informacao na cena e nao vale a pena rodar a YOLO. Aqui
medimos essa "novidade" com tecnicas classicas de OpenCV:

    frame -> escala de cinza -> suavizacao (blur) -> diferenca absoluta com o
    frame anterior -> threshold binario -> fracao de pixels que mudaram.

Se essa fracao passa de um limiar, consideramos que houve movimento relevante e
acionamos a deteccao.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TriggerResult:
    """Resultado da avaliacao de um frame pelo gatilho."""

    should_run: bool  # True -> rodar YOLO neste frame
    score: float  # fracao [0, 1] de pixels que mudaram em relacao ao frame anterior


class MotionTrigger:
    """Decide, frame a frame, se a inferencia deve ou nao ser executada.

    Parametros
    ----------
    diff_threshold:
        Limiar de intensidade (0-255) aplicado a diferenca absoluta entre frames.
        Diferencas menores que isso sao tratadas como ruido e zeradas.
    motion_threshold:
        Fracao da imagem (0-1) que precisa mudar para acionar a YOLO. Ex.: 0.005
        significa "0.5% dos pixels mudaram".
    blur_kernel:
        Tamanho (impar) do kernel do GaussianBlur. Suaviza o ruido do sensor que,
        sem tratamento, dispararia o gatilho a toa (ver extra do roteiro: zoom
        maximo revela o granulado da camera).
    use_grayscale:
        Se True, compara em 1 canal (luminancia) em vez de 3 (BGR). Mais rapido e
        menos sensivel a variacoes de cor.
    use_blur:
        Liga/desliga a suavizacao gaussiana (util para comparar o efeito do extra).
    """

    def __init__(
        self,
        diff_threshold: int = 25,
        motion_threshold: float = 0.005,
        blur_kernel: int = 5,
        use_grayscale: bool = True,
        use_blur: bool = True,
    ) -> None:
        self.diff_threshold = diff_threshold
        self.motion_threshold = motion_threshold
        # GaussianBlur exige kernel impar; corrige silenciosamente se vier par.
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self.use_grayscale = use_grayscale
        self.use_blur = use_blur
        self._prev: np.ndarray | None = None  # frame anterior ja pre-processado

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Converte para cinza e suaviza, conforme as opcoes ativas."""
        processed = frame
        if self.use_grayscale:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        if self.use_blur:
            k = self.blur_kernel
            processed = cv2.GaussianBlur(processed, (k, k), 0)
        return processed

    def evaluate(self, frame: np.ndarray) -> TriggerResult:
        """Avalia um frame e atualiza o estado interno (frame anterior).

        O primeiro frame sempre aciona, pois nao ha anterior para comparar.
        """
        current = self._preprocess(frame)

        if self._prev is None:
            self._prev = current
            return TriggerResult(should_run=True, score=1.0)

        # Diferenca absoluta pixel a pixel entre frame atual e anterior.
        diff = cv2.absdiff(current, self._prev)
        # Pixels com mudanca acima do limiar viram 255; o resto, 0.
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)

        # Sem grayscale a mascara tem 3 canais; colapsa para 1 ("mudou em
        # qualquer canal") para que countNonZero funcione.
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        changed = cv2.countNonZero(mask)
        total = mask.shape[0] * mask.shape[1]
        score = changed / total if total else 0.0

        self._prev = current
        return TriggerResult(should_run=score >= self.motion_threshold, score=score)
