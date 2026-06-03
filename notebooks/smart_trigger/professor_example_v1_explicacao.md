# Explicacao do notebook professor_example_v1.ipynb

Este arquivo resume, ponto a ponto, como o notebook executa o pipeline de deteccao + rastreamento.

## 1. Objetivo do exemplo

O notebook combina:
- YOLO para detectar objetos em keyframes (frames periodicos)
- Lucas-Kanade Optical Flow para propagar as caixas entre keyframes

Ideia: reduzir custo de inferencia, sem perder continuidade visual entre frames.

## 2. Imports

- `cv2`: leitura de video, processamento classico, desenho e exibicao
- `numpy`: calculos numericos (ex.: mediana de deslocamento)
- `YOLO` (ultralytics): detector de objetos

## 3. Inicio da funcao `main()`

### 3.1 Modelo YOLO

```python
model = YOLO("yolov8n.pt")
```

Carrega o modelo nano (mais leve e rapido).

### 3.2 Fonte de video

```python
cap = cv2.VideoCapture("videos/ref.mp4")
```

Abre arquivo de video. Para webcam, poderia ser `0`.

## 4. Parametros classicos

### 4.1 Shi-Tomasi (`feature_params`)

Controla como pontos fortes sao escolhidos dentro das caixas detectadas.

- `maxCorners=50`: no maximo 50 pontos por objeto
- `qualityLevel=0.1`: filtra cantos fracos
- `minDistance=7`: evita pontos muito proximos
- `blockSize=7`: tamanho da vizinhanca

### 4.2 Lucas-Kanade (`lk_params`)

Controla o optical flow:
- `winSize=(15, 15)`
- `maxLevel=2` (piramide)
- `criteria=(EPS|COUNT, 10, 0.03)`

## 5. Estado do pipeline

- `detect_interval = 15`: roda YOLO a cada 15 frames
- `threshold = 10`: minimo de pontos validos para manter um objeto
- `frame_idx = 0`: contador de frame
- `tracked_objects = []`: lista de objetos rastreados
- `prev_gray = None`: frame anterior em escala de cinza

Estrutura de cada item em `tracked_objects`:

```python
{
  'bbox': [x1, y1, x2, y2],
  'points': np.array
}
```

## 6. Loop principal

```python
while cap.isOpened():
```

A cada iteracao:
1. Le frame (`ret, frame = cap.read()`)
2. Se `ret` for falso, encerra
3. Converte para cinza (`gray`)
4. Copia frame para visualizacao (`vis_frame`)

## 7. Fase 1: Keyframe (YOLO)

Condicao:

```python
if frame_idx % detect_interval == 0 or not tracked_objects:
```

Quando entra aqui:
1. Limpa `tracked_objects`
2. Executa YOLO no frame atual
3. Extrai caixas `xyxy`
4. Para cada caixa:
   - cria mascara com a area da caixa
   - roda `goodFeaturesToTrack` dentro da mascara
   - se encontrou pontos, cria item em `tracked_objects`

Resultado: objetos detectados + pontos iniciais para rastreamento.

## 8. Fase 2: P-frame (Lucas-Kanade)

Quando NAO e keyframe:

1. Cria `new_tracked_objects`
2. Para cada objeto atual:
   - pega pontos antigos (`p0`) e bbox antiga
   - calcula optical flow com `calcOpticalFlowPyrLK(prev_gray, gray, p0, ...)`
   - filtra apenas pontos com status valido (`st == 1`)
3. Se sobrar menos que `threshold` pontos, descarta objeto
4. Senao:
   - calcula deslocamento robusto por mediana (`dx`, `dy`)
   - move bbox antiga por esse delta
   - salva bbox nova e pontos novos em `new_tracked_objects`
5. Substitui `tracked_objects = new_tracked_objects`

## 9. Visualizacao

Para cada objeto rastreado:
- desenha bbox verde
- desenha pontos vermelhos

Depois mostra o frame:

```python
cv2.imshow("YOLO Keyframe + LK Tracking", vis_frame)
```

## 10. Atualizacao de estado

No final de cada frame:
- `prev_gray = gray.copy()`
- `frame_idx += 1`
- se tecla `q` for pressionada, encerra

## 11. Encerramento

Ao sair do loop:
- `cap.release()`
- `cv2.destroyAllWindows()`

## 12. Comportamento esperado no tempo

- Frame inicial: YOLO detecta objetos e inicia pontos
- Frames intermediarios: LK propaga caixas sem nova inferencia
- A cada `detect_interval`: YOLO corrige/reinicializa o rastreamento

## 13. Pontos fortes

- Menor custo que rodar detector pesado em todos os frames
- Boa continuidade visual entre deteccoes
- Estrutura didatica para estudo de pipeline hibrido

## 14. Limitacoes

- Atualiza bbox por translacao (nao ajusta escala/rotacao)
- Pode falhar com oclusao forte ou mudanca brusca
- Nao faz persistencia de identidade robusta entre longos periodos
- Sem filtros de classe/confianca no exemplo atual

## 15. Diferenca para o Smart Trigger do projeto

No notebook atual, a decisao de rodar YOLO e por intervalo fixo (`detect_interval`).
No modulo `trigger.py`, a decisao e por mudanca entre frames (score de movimento).

Ou seja:
- notebook: agendamento fixo + rastreamento por LK
- pipeline smart_trigger: gatilho de movimento + detector sob demanda
