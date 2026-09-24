# OCR results

Small open models, same pages, this machine.
12th Gen Intel(R) Core(TM) i9-12900H, 31.0 GiB RAM, NVIDIA GeForce RTX 3060 Laptop GPU.
GPU memory in use when this run started: 2137 of 6144 MiB.
Generated 2026-09-24 19:07 UTC.

Character accuracy ignores HTML and LaTeX markup, and it ignores a tail the model repeated. It is averaged over pages that have a transcript.
Fields are the facts listed in `corpus/manifest.json`. A miss means that fact is not in the output.
Seconds are warm extraction, after the weights load.

| Model | Size | Result | Median s/page | Load s | Character accuracy | Fields |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| RapidOCR | PP-OCRv6 small | ok | 1.70 | 0.20 | 84% | 33/36 |
| OvisOCR2 | 0.8B | ok | 34.72 | 40.07 | 95% | 34/36 |
| PaddleOCR-VL-1.6 | 0.9B | ok | 47.66 | 12.93 | 93% | 29/36 |
| GLM-OCR | 0.9B | ok | 2.77 | 56.78 | 100% | 27/36 |
| LightOnOCR-2-1B | 1B | ok | 2.87 | 7.39 | 100% | 36/36 |
| TeleOCR | 1.2B | ok | 2.08 | 4.86 | 65% | 24/36 |

## Pages

### Clean letter

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 0.96 | 100% | 4/4 |
| OvisOCR2 | 36.83 | 97% | 4/4. Then the model repeated itself |
| PaddleOCR-VL-1.6 | 27.47 | 100% | 4/4 |
| GLM-OCR | 2.77 | 100% | 4/4 |
| LightOnOCR-2-1B | 2.29 | 100% | 4/4 |
| TeleOCR | 2.08 | 100% | 4/4 |

<details><summary>RapidOCR excerpt</summary>

```
Northwind Supply
14 March 2026
Dear Mina,
Please confirm the delivery for order 1842. The crates
should arrive at Dock 3 before 16:00. Two of the crates
are marked fragile.
Thank you,
Helena Costa
Accounts
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
Northwind Supply

14 March 2026

Dear Mina,

Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.

Thank you,

Helena Costa

Accounts

1

1

1
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
Northwind Supply
14 March 2026

Dear Mina,

Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.

Thank you,
Helena Costa
Accounts
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
Northwind Supply
14 March 2026

Dear Mina,

Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.

Thank you,
Helena Costa
Accounts
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
Northwind Supply  
14 March 2026

Dear Mina,

Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.

Thank you,  
Helena Costa  
Accounts
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
Northwind Supply
14 March 2026
Dear Mina,
Please confirm the delivery for order 1842. The crates should arrive at Dock 3 before 16:00. Two of the crates are marked fragile.
Thank you,
Helena Costa
Accounts
```

</details>

### Two columns and a footnote

Page 1 of corpus/paper.pdf. Reading order is the left column, then the right column, then the footnote.

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 1.70 | 55% | 4/5, missed Crate B holds oil |
| OvisOCR2 | 34.72 | 100% | 5/5. Then the model repeated itself |
| PaddleOCR-VL-1.6 | 47.66 | 74% | 4/5, missed left column ends here |
| GLM-OCR | 2.18 | 100% | 5/5 |
| LightOnOCR-2-1B | 2.49 | 100% | 5/5 |
| TeleOCR | 2.02 | 59% | 4/5, missed Order 1842 |

<details><summary>RapidOCR excerpt</summary>

```
Harbor Notes
Spare parts
The crane arrived on Tuesday.
Crate A holds bolts. Crate B holds
Order 1842 is still open. Dock 3
oil. Paint is stored in crate C.
closes at 16:00.
The right column ends here.
Inspectors counted the crates
twice. The left column ends here.
Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
Harbor Notes

The crane arrived on Tuesday.

Order 1842 is still open. Dock 3

closes at 16:00.

Inspectors counted the crates

twice. The left column ends here.

Spare parts

Crate A holds bolts. Crate B holds oil. Paint is stored in crate C.

The right column ends here.

Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
Harbor Notes
Spare parts

The crane arrived on Tuesday.
Order 1842 is still open. Dock 3
closes at 16:00.

Crate A holds bolts. Crate B holds
oil. Paint is stored in crate C.

The right column ends here.

Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
Harbor Notes

The crane arrived on Tuesday. Order 1842 is still open. Dock 3 closes at 16:00.

Inspectors counted the crates twice. The left column ends here.

Spare parts

Crate A holds bolts. Crate B holds oil. Paint is stored in crate C.

The right column ends here.

Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
Harbor Notes

The crane arrived on Tuesday.  
Order 1842 is still open. Dock 3 closes at 16:00.

Inspectors counted the crates twice. The left column ends here.

Spare parts

Crate A holds bolts. Crate B holds oil. Paint is stored in crate C.

The right column ends here.

---

Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
Harbor Notes Spare parts
The crane arrived on Tuesday. Crate A holds bolts. Crate B holds oil. Paint is stored in crate C.
C
The right column ends here.
Inspectors counted the crates twice. The left column ends here.
# Note 1. The tide turns at 18:40, so the barge must leave before then.
```

</details>

### Table and an equation

Page 2 of corpus/paper.pdf.

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 1.01 | 96% | 4/4 |
| OvisOCR2 | 35.79 | 81% | 4/4. Then the model repeated itself |
| PaddleOCR-VL-1.6 | 31.21 | 100% | 4/4 |
| GLM-OCR | 1.28 | 100% | 4/4 |
| LightOnOCR-2-1B | 2.37 | 100% | 4/4 |
| TeleOCR | 1.54 | 0% | 0/4, missed Bolts, 2.5, 12.0, 136.0 |

<details><summary>RapidOCR excerpt</summary>

```
Load table
Item
Weight kg
Count
Bolts
2.5
40
l!O
12.0
3
w = 2.5 * 40 + 12.0 * 3 = 136.0 kg
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
Load table

<table border="1"><tr><td>Item</td><td>Weight kg</td><td>Count</td></tr><tr><td>Bolts</td><td>2.5</td><td>40</td></tr><tr><td>Oil</td><td>12.0</td><td>3</td></tr></table>

$$ w=2.5\times40+12.0\times3=136.0\ \text{kg} $$
 code
 code
 code
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
Load table

Item          Weight kg          Count
Bolts          2.5                40
Oil                12.0              3

w = 2.5 * 40 + 12.0 * 3 = 136.0 kg
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
Load table

| Item | Weight kg | Count |
| :--- | :--- | :--- |
| Bolts | 2.5 | 40 |
| Oil | 12.0 | 3 |

$w = 2.5 * 40 + 12.0 * 3 = 136.0 \text{ kg}$
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
# Load table

<table>
  <tr>
    <th>Item</th>
    <th>Weight kg</th>
    <th>Count</th>
  </tr>
  <tr>
    <td>Bolts</td>
    <td>2.5</td>
    <td>40</td>
  </tr>
  <tr>
    <td>Oil</td>
    <td>12.0</td>
    <td>3</td>
  </tr>
</table>

$$
w = 2.5 * 40 + 12.0 * 3 = 136.0 \text{ kg}
$$
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
<box:043 045 228 049><label:title><up>
<box:044 113 855 301><label:table><up>
<box:038 406 595 443><label:text><up>
```

</details>

### Tilted, faded receipt

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 1.03 | 84% | 5/5 |
| OvisOCR2 | 34.26 | 100% | 5/5. Then the model repeated itself |
| PaddleOCR-VL-1.6 | 30.01 | 100% | 5/5 |
| GLM-OCR | 1.06 | 100% | 5/5 |
| LightOnOCR-2-1B | 2.87 | 100% | 5/5 |
| TeleOCR | 1.50 | 100% | 5/5 |

<details><summary>RapidOCR excerpt</summary>

```
NORTHWIND MARKET
14 March 2026
12:41
3.40
Oat milk
4.50
Sourdough
10.50
Coffee beans
18.40
Subtotal
0.00
Tax
18.40
Total
Card payment
Thank you
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
NORTHWIND MARKET

14 March 2026

12:41

Oat milk

3.40

Sourdough

4.50

Coffee beans

10.50

Subtotal

18.40

Tax

0.00

Total

18.40

Card payment

Thank you
<think>
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
NORTHWIND MARKET
14 March 2026
12:41
Oat milk 3.40
Sourdough 4.50
Coffee beans 10.50
Subtotal 18.40
Tax 0.00
Total 18.40
Card payment
Thank you
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
NORTHWIND MARKET
14 March 2026
12:41

Oat milk 3.40
Sourdough 4.50
Coffee beans 10.50

Subtotal 18.40
Tax 0.00
Total 18.40

Card payment
Thank you
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
NORTHWIND MARKET  
14 March 2026  
12:41

<table>
  <tr>
    <td>Oat milk</td>
    <td>3.40</td>
  </tr>
  <tr>
    <td>Sourdough</td>
    <td>4.50</td>
  </tr>
  <tr>
    <td>Coffee beans</td>
    <td>10.50</td>
  </tr>
  <tr>
    <td>Subtotal</td>
    <td>18.40</td>
  </tr>
  <tr>
    <td>Tax</td>
    <td>0.00</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>18.40</td>
  </tr>
</table>

Card payment  
Thank you
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
NORTHWIND MARKET
14 March 2026
12:41
Oat milk 3.40
Sourdough 4.50
Coffee beans 10.50
Subtotal 18.40
Tax 0.00
Total 18.40
Card payment
Thank you
```

</details>

### Invoice tax and product table

Real NF-e block. Scored on the filled-in amounts, the product code, NCM, CFOP, freight payer, weight, and the two retained-tax lines. The 0,00 cells are not required.

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 2.42 | — | 8/10, missed NOTEBOOK, 84713019 |
| OvisOCR2 | 31.97 | — | 9/10, missed 5405 |
| PaddleOCR-VL-1.6 | 91.20 | — | 7/10, missed 3.254,07, 4.000, Emitente |
| GLM-OCR | 19.56 | — | 1/10, missed 5763764, NOTEBOOK, 84713019, 5405, 1.0000, 4.000, Emitente, 2821,85, 117,42 |
| LightOnOCR-2-1B | 18.43 | — | 10/10 |
| TeleOCR | 26.20 | — | 6/10, missed 84713019, 5405, 1.0000, 2821,85 |

<details><summary>RapidOCR excerpt</summary>

```
CÁLCULO DO IMPOSTO
BASE DE CÁLCULO DO ICMS
VALOR DO ICMS
BASE DE CÁLCULO DO ICMS ST VALOR DO ICMS ST
VALOR APROXIMADO DOS TRIBUTOS
VALOR TOTAL DOS PRODUTOS
0,00
0,00
0,00
3.254,07
VALOR DO FRETE
VALOR DO SEGURO
DESCONTO
OUTRAS DESPESAS ACESSÓRIAS
VALOR DO IPI
VALOR TOTAL DA NOTA
0,00
0,00
0,00
3.254,07
TRANSPORTADOR / VOLUMES TRANSPORTADOS DADOS
RAZÃO SOCIAL
FRETE POR CONTA
CÓDIGO ANTT
PLACA DO VEÍCULO
UF
CNPJ / CPF
0
Emitente
ENDEREÇO
MUNICÍPIO
UF
INSCRIÇÃO ESTADUAL
QUANTIDADE
ESPÉCIE
MARCA
NUM
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
CALCULO DO IMPOSTO

<table border=1><tr><td colspan="2">BASE DE CÁLCULO DO ICMS</td><td colspan="2">VALOR DO ICMS</td><td>0,00</td><td>BASE DE CÁLCULO DO ICMS ST 0,00</td><td>VALOR DO ICMS ST 0,00</td><td>VALOR APROXIMADO DOS TRIBUTOS 0,00</td><td>VALOR TOTAL DOS PRODUTOS 3.254,07</td></tr><tr><td>VALOR DO FRETE 0,00</td><td>VALOR DO SEGURO 0,00</td><td>0,00</td><td>DESCONTO 0,00</td><td>OUTRAS DESPESAS ACESSÓRIAS 0,00</td><td>VALOR DO IPI 0,00</td><td>VALOR TOTAL DA NOTA 3.254,07</td></tr></tab
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
CALCULO DO IMPOSTO
BASE DE CÁLCULO DO ICMS
0,00
VALOR DO ICMS
0,00
BASE DE CÁLCULO DO ICMS ST
0,00
VALOR DO ICMS ST
0,00
VALOR APROXIMADO DOS TRIBUTOS
0,00
VALOR TOTAL DOS PRODUTOS
3,254,07
VALOR DO FRETE
0,00
VALOR DO SEGURO
0,00
DESCONTO
0,00
OUTRAS DESPESAS ACESSÓRIAS
0,00
VALOR DO IPI
0,00
VALOR TOTAL DA NOTA
3,254,07
TRANSPORTADOR / VOLUMES TRANSPORTADOS DADOS
RAZÃO SOCIAL
ENDEREÇO
QUANTIDADE 1 ESPÉCIE VOLUMES MARCA NUMIÇIPIO PESO BRUTO INSCRIÇÃO ESTADUAL
DADOS DO PRODUTO / SERVIÇO
COD.PROD
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
CALCULO DO IMPOSTO
BASE DE CÁLCULO DO ICMS
0,00
VALOR DO ICMS
0,00
BASE DE CÁLCULO DO ICMS ST
0,00
VALOR DO ICMS ST
0,00
VALOR APROXIMADO DOS TRIBUTOS
0,00
VALOR TOTAL DOS PRODUTOS
3.254,07
VALOR DO FRETE
0,00
VALOR DO SEGURO
0,00
DESCONTO
0,00
OUTRAS DESPESAS ACESSÓRIAS
0,00
VALOR DO IPI
0,00
VALOR TOTAL DA NOTA
3.254,07
TRANSPORTADOR / VOLUMES TRANSPORTADOS DADOS
RAZÃO SOCIAL
FRETE POR CONTA
CÓDIGO ANTT
PLACA DO VEÍCULO
UF
CNPJ / CPF
ENDEREÇO
MARCA
MUNICIPIO
UF
INSCRIÃO ESTADUAL
ENDEREÇO
ESPÉC
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
CÁLCULO DO IMPOSTO

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>BASE DE CÁLCULO DO ICMS</th>
      <th>VALOR DO ICMS</th>
      <th>BASE DE CÁLCULO DO ICMS ST</th>
      <th>VALOR DO ICMS ST</th>
      <th>VALOR APROXIMADO DOS TRIBUTOS</th>
      <th>VALOR TOTAL DOS PRODUTOS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0,00</td>
      <td>0,00</td>
      <td>0,00</td>
      <td>3.254,07</td>
    </tr>
  </tbody>
</table>

<table border="
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
CALCULO DO IMPOSTO

BASE DE CÁLCULO DO ICMS 0,00 VALOR DO ICMS 0,00 BASE DE CÁLCULO DO ICMS ST 0,00 VALOR DO ICMS ST 0,00 VALOR APROXIMADO DOS TRIBUTOS 0,00 VALOR TOTAL DOS PRODUTOS 3.254,07
VALOR DO FRETE 0,00 VALOR DO SEGURO 0,00 DESCONTO 0,00 OUTRAS DESPESAS ACESSÓRIAS 0,00 VALOR DO IPI 0,00 VALOR TOTAL DA NOTA 3.254,07
TRANSPORTADOR / VOLUMES TRANSPORTADOS DADOS

RAZÃO SOCIAL
FRETE POR CONTA 0 - Emitente CÓDIGO ANTT PLACA DO VEÍCULO UF CNPJ / CPF
ENDEREÇO MUNICIPIO UFR INSCRIÇÃO
QUANTIDADE 1
```

</details>

### Phone photo of a package

Real photo. Scored on the readable nutrition lines. The tiny print has no full transcript.

| Model | Seconds | Character accuracy | Fields |
| --- | ---: | ---: | --- |
| RapidOCR | 2.00 | — | 8/8 |
| OvisOCR2 | 31.96 | — | 7/8, missed nao contem gluten |
| PaddleOCR-VL-1.6 | 1132.59 | — | 5/8, missed carboidratos, 375, nao contem gluten |
| GLM-OCR | 29.75 | — | 8/8 |
| LightOnOCR-2-1B | 26.53 | — | 8/8 |
| TeleOCR | 29.95 | — | 5/8, missed Nestle, CHOCOLATERIA, nao contem gluten |

<details><summary>RapidOCR excerpt</summary>

```
Nestle
CHOCOLATERIA
0800-770241-www.nestle.com.br
Serviçe Nestle ao Consumidor
NUTRITIONAL COMPASSS
açucar e aromatizante
Ingredientes: cacau em pé
Nestie
NÃO CONTEM GLUTEN
Faz Bem
en polvo, azicary
Ingredientes: Cacao
INFORMAÇÃO NUTRICIONAL
NO CONTIENE GLUTEN
aromatizante artificial
Porções por embalagem: 10
20g
%VD*
Porção: 20 g (2 Colheres de sopa)
100g
73
4
4
375
57
11
50
10
10
20
Valor energético (kcal)
50
2,3
2
5
Carboidratos (g)
Açúcares totais (g)
6.5
12
0.7
1.3
4
Açúcares adicionados (g
```

</details>

<details><summary>OvisOCR2 excerpt</summary>

```
Nestlé

CHOCGLATERIA

Nestlé.

Faz Bem

Grua de Nestlé CHOCOLATERIA

De todas as

Servicio Nestlé al Consumidor

0800-7700411 - www.nestlé.com.br
```

</details>

<details><summary>PaddleOCR-VL-1.6 excerpt</summary>

```
Nestlé
CHOCOLATERIA

Porções por embalagem: 10 porção: 20 g (2 Colheres de sopa)
Valor energético (kcal)
Carbohidratos (g)
Áquies totais (g)
Proteínas (g)
Gorduras saturadas (g)
Gorduras trans (g)
Fibras alimentares (g)
Sodio (mg)
Percentual de valores diários fornecidos pela porção
Carbohidratos 11 g (2%VD) Grasas saturadas 10 g (2%VD)
Carbohidratos 11 g (1%VD) de los cuales Azúcares fobras 10 g (2%VD)
(3%VD) Grasas totais 1.3 g (2%VD) Grasas saturadas 10.7 g (2%VD)
(14%VD) Sodio 1.9 mg (0%VD)
```

</details>

<details><summary>GLM-OCR excerpt</summary>

```
Nestle
CHOCOLATERIA
Oma dua NESTLE® CHOCOLATERIA
Re admissões feitas com chocolate são sublimadas a preparação em
momento algerite e posse de azúcares. Consumir com mordre
a chocolata pode fazer para de uma alimentação equilibrada
Serviço Nestle ao Consumidor
0800-7702411 - www.nestle.com.br
Serviço Nestle al Consumidor
PY: 0800 11 2121 - www.nestle.com.py
NUTRITIONAL COMPASS®
© Marca Registrada de Société des Produits Nestlé S.A.
INFORMAÇÃO NUTRICIONAL
Porções por embalagem: 10
Porção: 20 g (2
```

</details>

<details><summary>LightOnOCR-2-1B excerpt</summary>

```
# Nestlé

## CHOCOLATERIA

Nesta abra NESTLÉ® CHOCOLATENIA  
No entramos telem como chocolate sé subreto y proporcional  
porcino alégra e perto apela as relações. Consumida com mante  
chocolato perto bão, perto de uma alimentação equilíbrio

Serviço Nestlé ao Consumidor  
0800-7705411 - www.nestle.com.br  
Serviço Nestlé al Consumidor  
0800 11 2121 - www.nestle.com.py  
NUTRITIONAL COMPASS®  
®Marca Registrada de Sociedade de Nutrição

---

**Nestlé**  
Faz Beim

---

## INFORMAÇÃO NUTRICIONA
```

</details>

<details><summary>TeleOCR excerpt</summary>

```
INFORMAÇÃO NUTRICIONAL
Porções por embalagem: 10
Porção: 20 g (2 Colheres de sopa)
Valor energético (kcal) 100 g 20 g %VD*
Carboidratos (g) 375 73 4
Açúcares totais (g) 57 11 4
Açúcares adicionados (g) 50 10 20
Proteinas (g) 12 2.3 5
Gorduras totais (g) 6.5 1.3 2
Gorduras saturadas (g) 3.3 0.7 4
Fibras aliments (g) 0 0 0
Sodic (mg) 18 3.5 14
Percentile of gondrode (gondrode) 100 g 20 g 20 g
Incentrode (20%) of gondrode (20%) of gondrode (20%) of gondrode (20%) of gondrode (20%) of gondrode (20%)
```

</details>

## Notes

### RapidOCR

ONNX Runtime on CPU. The installed wheel ships the small PP-OCRv6 detection and recognition models.

### OvisOCR2

End-to-end. One image in, Markdown out. Greedy decoding, 2048 new tokens.

### PaddleOCR-VL-1.6

Single prompt OCR:, not Paddle's layout pipeline. The 1.6 checkpoint leaves out vision rope settings, so the runner fills those in before load.

### GLM-OCR

Checkpoint only. The published page score uses the GLM-OCR SDK, which adds PP-DocLayout-V3 before recognition.

### LightOnOCR-2-1B

End-to-end. No text prompt, matching the model card. Greedy decoding, 2048 new tokens.

### TeleOCR

Text prompt from the model card. This run uses transformers 4.57.1, which matches the model's own code. Newer transformers does not load it.

Re-run with `python run.py` from the repo root.
