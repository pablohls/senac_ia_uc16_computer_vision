# Como funciona o Smart Inference Trigger

Documento explicativo do projeto: o que cada parte faz, como a detecção é
realizada e por que isso reduz o custo computacional.

---

## Visão geral: dois "cérebros" trabalhando juntos

O sistema tem duas etapas com papéis bem diferentes:

| | **Gatilho** (`trigger.py`) | **Detector** (`detector.py`) |
|---|---|---|
| O que faz | decide *se* vale olhar | descobre *o que* tem na imagem |
| Técnica | CV clássica (matemática simples) | rede neural (YOLO) |
| Custo | baixíssimo (~microssegundos) | alto (a parte cara) |
| Pergunta | "mudou alguma coisa?" | "quais objetos, onde, e o quê?" |

O `main.py` coordena os dois: para cada frame, pergunta ao gatilho e só chama o
detector quando vale a pena.

---

## O fluxo, frame a frame

```
   vídeo / webcam / smartphone
              │
              ▼
        ┌───────────┐
        │ lê 1 frame│
        └─────┬─────┘
              ▼
   ┌─────────────────────────┐
   │ GATILHO (CV clássica)    │   barato
   │ "mudou o suficiente?"    │
   └───────┬──────────┬───────┘
        SIM│          │NÃO
           ▼          ▼
   ┌────────────┐  ┌────────────────────┐
   │ roda a YOLO│  │ reaproveita as      │
   │ (detecta)  │  │ últimas caixas      │
   │  caro      │  │ (cache, custo zero) │
   └─────┬──────┘  └──────────┬─────────┘
         └──────────┬─────────┘
                    ▼
        ┌────────────────────────┐
        │ desenha status na tela  │
        │ + atualiza estatísticas │
        └────────────────────────┘
                    │
                    ▼  (próximo frame)
```

---

## Parte 1 — Como a detecção (YOLO) funciona

### Classificação vs. detecção

- **Classificação**: "esta imagem é um gato" — uma resposta para a imagem toda.
- **Detecção**: "há uma *pessoa* neste retângulo (92% de certeza) e um *celular*
  naquele outro" — **vários objetos**, cada um com **caixa (bounding box) +
  classe + confiança**.

Nosso projeto faz **detecção**.

### O que é a YOLO

YOLO = *You Only Look Once* ("você só olha uma vez"). Em vez de varrer a imagem
várias vezes procurando objetos pedaço por pedaço, ela faz **uma única passada**
pela rede neural e produz todas as caixas de uma vez. Por isso é rápida o
bastante para vídeo.

### O que acontece quando chamamos `self.model(frame)`

Em `detector.py`, a linha `results = self.model(frame, verbose=False)` dispara:

1. **Pré-processamento** — o frame é redimensionado (ex.: 640×640) e normalizado
   (pixels 0–255 viram 0–1).
2. **Backbone (CNN)** — uma rede convolucional extrai *características*: as
   primeiras camadas captam bordas e texturas; as profundas, formas complexas
   (rodas, rostos, contorno de pessoa). É a "visão" da rede.
3. **Neck + Head** — combinam essas características em várias escalas (para achar
   objetos grandes e pequenos) e geram **milhares de caixas candidatas**, cada uma
   com coordenadas (x, y, largura, altura), uma pontuação de "tem objeto aqui?" e
   as probabilidades de cada classe.
4. **NMS (Non-Maximum Suppression)** — o mesmo objeto gera muitas caixas
   sobrepostas; o NMS mantém só a melhor de cada grupo. Por isso, no fim, sobra
   **uma caixa por objeto**.

Do resultado (`results[0]`) usamos:
- `result.boxes` — a lista de detecções (caixas + classe + confiança);
  `len(result.boxes)` é quantos objetos foram encontrados;
- `result.plot()` — desenha as caixas e rótulos sobre uma cópia do frame (o que
  vai para a tela).

### De onde vêm os rótulos "person", "cell phone"...

Usamos `yolov8m.pt`, um modelo **pré-treinado** no dataset **COCO**, que tem **80
classes** (pessoa, carro, cachorro, celular, cadeira...). Ele já aprendeu essas
categorias a partir de centenas de milhares de imagens rotuladas — nós não
treinamos nada, apenas carregamos esse conhecimento pronto. O sufixo `m` é o
tamanho "médio": mais preciso que o `n`/`s` e mais rápido que o `l`/`x` — o "meio
do caminho" que o roteiro pede.

---

## Parte 2 — Como o gatilho funciona

O gatilho responde "mudou alguma coisa?" com 5 operações baratas de OpenCV
(`trigger.py`):

1. **Cinza** (`cvtColor`) — compara em 1 canal (brilho) em vez de 3 (BGR): mais
   rápido e menos sensível a variação de cor.
2. **Blur** (`GaussianBlur`) — suaviza para apagar o ruído do sensor (aquele
   chuvisco que aparece no zoom máximo), que senão dispararia o gatilho à toa.
3. **Diferença** (`absdiff`) — subtrai o frame atual do anterior; onde nada mudou
   fica preto, onde mudou fica claro.
4. **Threshold** (`threshold`) — converte a diferença numa máscara branco/preto:
   branco = mudou de verdade; preto = ruído desprezível.
5. **Pontuação** — fração de pixels brancos (`countNonZero / total`). Se passa do
   limiar (`motion_threshold`), há movimento relevante → vale rodar a YOLO.

Repare: o gatilho **não sabe o que** está na cena, e nem precisa. Ele só mede a
*quantidade de mudança* — por isso é tão barato comparado à YOLO.

---

## Parte 3 — Como os dois se conectam (`main.py`)

Para cada frame:

```
frame → trigger.evaluate(frame)
          ├─ should_run = True  → detector.detect(frame)       (roda a YOLO, atualiza o cache)
          └─ should_run = False → detector.last_annotated(...)  (reaproveita as últimas caixas)
       → desenha status na tela e contabiliza estatísticas
```

O **cache** (`_last_annotated` em `detector.py`) é o que mantém as caixas na tela
mesmo nos frames pulados: em vez de inferir de novo, mostra o último resultado
válido. Como a cena não mudou, esse resultado ainda vale — é exatamente a premissa
do projeto.

---

## Por que isso economiza

A YOLO custa, por exemplo, ~50 ms por frame; o gatilho custa fração de
milissegundo. Em uma câmera fixa onde **boa parte do tempo nada acontece**, o
gatilho pula esses trechos e a YOLO só roda quando há movimento.

Nos testes, isso significou **pular ~63% dos frames** — ou seja, ~63% da carga da
YOLO economizada, sem perder detecções relevantes. Multiplicando por dezenas de
câmeras (o cenário do prédio citado no enunciado), essa economia pode ser a
diferença entre rodar localmente e precisar de uma solução em nuvem.

Para medir essa economia no projeto, compare os dois modos:

```bash
poetry run smart-trigger --source videos/ref.mp4 --baseline --no-show  # YOLO em todo frame
poetry run smart-trigger --source videos/ref.mp4 --no-show             # com o gatilho
```

(ou rode a seção 5 do `smart_trigger_demo.ipynb`).

---

## Próximo passo: Lucas-Kanade (extensão futura)

O gatilho atual mede *quanto* mudou, mas não *o que* nem *para onde*. Um
refinamento é o **optical flow de Lucas-Kanade** (`cv2.calcOpticalFlowPyrLK`), que
rastreia pontos-chave entre frames e estima o vetor de movimento de cada um. Isso
permite distinguir movimento coerente (um objeto se deslocando) de ruído ou
variação de iluminação espalhada, tornando o gatilho mais robusto.
