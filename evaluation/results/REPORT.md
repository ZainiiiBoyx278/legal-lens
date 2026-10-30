# Historical IPC retrieval and scope evaluation

> Historical research material only. This is not current-law advice, legal applicability, or a legal correctness benchmark.

- Corpus SHA-256: `5706a1b995df774c4c4ea1868223e18a13ba619977d323d3cab76a1cc095e237`
- Executed (UTC): 2026-10-30T09:42:21.131749+00:00
- Source: Indian Penal Code Book (1871 physical PDF pages)
- Model: `sentence-transformers/all-MiniLM-L6-v2`; chunking: page-aware-1100-140-v1

**Label quality:** Author-created, source-checked pilot questions. They are not sampled from real users, independently adjudicated, or validated by legal experts. Labels contain selected physical pages and exact section markers; other relevant pages may be unlabelled.

**What is measured:** Hit@5 and MRR@5 count retrieval of a labelled page *and* matching section marker. This is an evidence-location proxy, not proof that an answer is correct. Latency is warm in-process wall-clock time on this machine; it excludes one-time indexing/model loading and is not a deployment SLA.

The semantic-only method is an evaluation baseline using the same local MiniLM encoder and threshold, not an existing standalone product feature. BM25 and hybrid call the actual application service. Hybrid may use a BM25 fallback; fallbacks are shown below but excluded from hybrid metrics and their denominator.

| Method | Label hit@5 | MRR@5 | p50 ms | p95 ms | Fallbacks / unavailable |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25 | 14/27 (51.9%) | 0.265 | 14.03 | 17.91 | 0 / 0 |
| semantic | 10/27 (37.0%) | 0.140 | 12.19 | 14.29 | 0 / 0 |
| hybrid | 14/27 (51.9%) | 0.251 | 26.64 | 31.76 | 0 / 0 |

### Pilot breakdown

| Question type | BM25 hits | Semantic hits | Hybrid hits |
| --- | ---: | ---: | ---: |
| Explicit section lookup | 10/21 | 9/21 | 12/21 |
| Paraphrase | 4/6 | 1/6 | 2/6 |

## Supported historical questions

A hit means the labelled marker is found in the top five retrieved passages. A miss is not automatically a wrong legal answer.

| ID | Question | Gold section / pages | BM25 | Semantic baseline | Hybrid |
| --- | --- | --- | --- | --- | --- |
| h01 | In this historical book, what does section 24 say about acting dishonestly? | 24 / 55 | hit rank 2; 15.18 ms; pages 55, 55, 1695, 661, 862 | hit rank 2; 10.86 ms; pages 55, 55, 55, 229 | hit rank 2; 31.08 ms; pages 55, 55, 55, 1695, 229 |
| h02 | In the historical IPC, what does section 34 say about common intention? | 34 / 75, 93 | miss; 15.0 ms; pages 77, 95, 82, 101, 100 | miss; 10.13 ms; pages 95, 82, 79, 78, 79 | miss; 21.87 ms; pages 82, 95, 96, 101, 101 |
| h03 | What does section 96 say about private defence in the historical IPC? | 96 / 320 | miss; 15.11 ms; pages 86, 103, 322, 325, 321 | hit rank 5; 13.25 ms; pages 321, 325, 322, 323, 320 | hit rank 5; 23.15 ms; pages 321, 325, 322, 86, 320 |
| h04 | What does the historical IPC state in section 299 about culpable homicide? | 299 / 1012 | miss; 15.99 ms; pages 1132, 1024, 1104, 1026, 1016 | miss; 14.62 ms; pages 1026, 1026, 1104, 1026, 1106 | miss; 28.29 ms; pages 1104, 1132, 1024, 1026, 1026 |
| h05 | What does section 300 say about murder in the historical IPC? | 300 / 1021 | miss; 14.2 ms; pages 1026, 1104, 1023, 1013, 1029 | miss; 12.82 ms; pages 1026, 1026, 1104, 1460, 1459 | miss; 26.64 ms; pages 1026, 1104, 1026, 1104, 1026 |
| h06 | What punishment does section 302 describe in the historical IPC? | 302 / 1077 | hit rank 2; 13.18 ms; pages 1095, 1077, 1460, 569, 568 | miss; 12.68 ms; pages 1095, 1460, 1459, 100, 1039 | miss; 25.62 ms; pages 1460, 1095, 1095, 569, 100 |
| h07 | What does section 304 say about culpable homicide not amounting to murder in this historical book? | 304 / 1103 | miss; 17.91 ms; pages 1132, 1014, 1104, 273, 1104 | miss; 11.11 ms; pages 1132, 1026, 1026, 1024, 1104 | miss; 26.27 ms; pages 1132, 1104, 1024, 1025, 1026 |
| h08 | What is hurt according to section 319 of the historical IPC? | 319 / 1199 | hit rank 4; 11.85 ms; pages 1203, 1838, 1199, 1199, 1199 | miss; 11.46 ms; pages none | hit rank 4; 29.07 ms; pages 1203, 1838, 1199, 1199, 1199 |
| h09 | Which kinds of injury are designated grievous hurt in section 320 of the historical IPC? | 320 / 1201 | hit rank 2; 15.79 ms; pages 1201, 1201, 1204, 1202, 1203 | hit rank 3; 13.93 ms; pages 1201, 1247, 1201, 1202, 1203 | hit rank 2; 28.21 ms; pages 1201, 1201, 1203, 1247, 1204 |
| h10 | What does section 322 of the historical IPC say about voluntarily causing grievous hurt? | 322 / 1207 | hit rank 2; 17.91 ms; pages 1839, 1207 | hit rank 2; 13.86 ms; pages 1839, 1207 | hit rank 2; 32.05 ms; pages 1839, 1207 |
| h11 | In the historical IPC what does section 323 state about punishment for voluntarily causing hurt? | 323 / 1208 | hit rank 1; 18.83 ms; pages 1208, 1215, 1248, 117, 1222 | hit rank 2; 14.39 ms; pages 1177, 1208, 1215, 1210, 1209 | hit rank 1; 26.72 ms; pages 1208, 1215, 1177, 1248, 117 |
| h12 | What does the historical IPC section 378 say about theft? | 378 / 1407 | miss; 12.86 ms; pages 1409, 1413, 1419, 1413, 1419 | hit rank 2; 12.84 ms; pages 1409, 1407, 1413 | hit rank 3; 22.92 ms; pages 1409, 1413, 1407, 1413, 1419 |
| h13 | What punishment for theft is described in section 379 of the historical IPC? | 379 / 1410 | miss; 12.92 ms; pages 1413, 1419, 1413, 1412, 1414 | miss; 14.05 ms; pages 1423 | miss; 26.98 ms; pages 1423, 1413, 1419, 1413, 1412 |
| h14 | What does section 390 of the historical IPC say about robbery? | 390 / 1440 | hit rank 3; 12.0 ms; pages 1850, 1441, 1440, 1441, 1442 | miss; 11.59 ms; pages 1441, 1442, 1442, 1850 | hit rank 5; 21.88 ms; pages 1441, 1850, 1442, 1442, 1440 |
| h15 | What does section 405 of the historical IPC describe as criminal breach of trust? | 405 / 1483 | miss; 13.88 ms; pages 1503, 1484, 1488, 1488, 1852 | hit rank 5; 12.19 ms; pages 1503, 1484, 1484, 1488, 1483 | hit rank 5; 22.24 ms; pages 1503, 1484, 1488, 1488, 1483 |
| h16 | What is cheating in section 415 of the historical IPC? | 415 / 1527 | miss; 12.12 ms; pages 1532, 1529, 1544, 1534, 1536 | miss; 12.7 ms; pages 1532, 1536, 1548, 1544, 1532 | miss; 18.98 ms; pages 1532, 1536, 1544, 1548, 1532 |
| h17 | What does section 420 say about cheating and delivery of property in the historical IPC? | 420 / 1547 | hit rank 2; 17.27 ms; pages 1534, 1547, 1542, 1531, 1548 | hit rank 3; 13.39 ms; pages 1548, 55, 1547, 1532, 203 | hit rank 1; 30.84 ms; pages 1547, 1548, 1534, 1542, 55 |
| h18 | In the historical IPC what is criminal trespass under section 441? | 441 / 1585 | hit rank 2; 9.01 ms; pages 1588, 1585, 1591, 1587, 1587 | miss; 11.3 ms; pages 1607, 1591, 1586, 1587, 1857 | hit rank 4; 26.57 ms; pages 1591, 1607, 1587, 1585, 1587 |
| h19 | What does the historical IPC say about defamation in section 499? | 499 / 1729 | miss; 13.31 ms; pages 1750, 1732, 1733, 1749, 1752 | miss; 11.05 ms; pages 1746, 1752, 779, 1746, 1748 | miss; 23.36 ms; pages 1752, 1732, 779, 1750, 1746 |
| h20 | What does the historical IPC say about criminal intimidation in section 503? | 503 / 1764 | hit rank 2; 15.59 ms; pages 1765, 1764, 703, 1868, 1765 | hit rank 2; 13.59 ms; pages 1868, 1764, 1765 | hit rank 2; 29.29 ms; pages 1765, 1764, 1868, 703, 1765 |
| h21 | What does section 511 of the historical IPC say about attempts? | 511 / 1780 | miss; 12.94 ms; pages 1836, 1780, 425, 180, 399 | miss; 10.69 ms; pages 1783, 1784 | miss; 25.01 ms; pages 1784, 1783, 1836, 1780, 425 |
| p01 | In the old IPC book, where does it define taking movable property without consent? | 378 / 1407 | hit rank 4; 12.21 ms; pages 1409, 1476, 1419, 1407, 1854 | miss; 10.75 ms; pages 51, 1583, 1484, 483, 1634 | miss; 31.06 ms; pages 1409, 51, 1417, 1488, 794 |
| p02 | In the old IPC book, where does it define causing bodily pain or disease? | 319 / 1199 | hit rank 2; 14.03 ms; pages 1199, 1199, 1202, 1244, 1838 | hit rank 5; 10.79 ms; pages 1018, 754, 1244, 1177, 1199 | hit rank 1; 26.71 ms; pages 1199, 1244, 1199, 1202, 1838 |
| p03 | In the old IPC book, where does it discuss threatening injury to reputation or property? | 503 / 1764 | miss; 11.92 ms; pages 1764, 731, 130, 1533, 1541 | miss; 9.23 ms; pages 1649, 590, 598, 754, 1583 | miss; 23.32 ms; pages 591, 599, 1509, 1753, 1564 |
| p04 | In the old IPC book, where does it discuss dishonest misappropriation of entrusted property? | 405 / 1483 | miss; 11.64 ms; pages 1478, 1488, 1476, 1484, 1476 | miss; 12.73 ms; pages 1560, 1484, 590, 598, 1486 | miss; 34.5 ms; pages 1486, 1484, 1476, 1477, 1476 |
| p05 | In the old IPC book, where does it describe deception that induces delivery of property? | 415 / 1527 | hit rank 3; 16.88 ms; pages 1547, 1431, 1527, 1527, 979 | miss; 11.96 ms; pages 1536, 1548, 1484, 1528, 754 | miss; 25.16 ms; pages 1528, 1528, 63, 1541, 1854 |
| p06 | In the old IPC book, where does it describe acting to cause wrongful gain or wrongful loss? | 24 / 55 | hit rank 1; 16.75 ms; pages 55, 53, 53, 1634, 55 | miss; 10.93 ms; pages 72, 754, 53, 1566, 1649 | hit rank 3; 27.44 ms; pages 53, 53, 55, 1566, 55 |

## Out-of-scope questions

These test the actual chat service in provider-free excerpt mode. A pass means the service returned `no_evidence`; it does not establish legal correctness.

| ID | Question | Expected | Actual | Outcome | Reason | ms |
| --- | --- | --- | --- | --- | --- | ---: |
| u01 | What is the current punishment under BNS for murder? | no_evidence | no_evidence | pass | This question mentions BNS/BNSS/BSA material that is not verified in the selected source. Selected source status: historical. | 10.03 |
| u02 | Compare IPC section 302 with BNS section 103. | no_evidence | no_evidence | pass | This question mentions BNS/BNSS/BSA material that is not verified in the selected source. Selected source status: historical. | 8.92 |
| u03 | What is the latest criminal law for theft in India? | no_evidence | no_evidence | pass | This question asks about current law or present applicability. Selected source status: historical. | 9.38 |
| u04 | Is this provision still in force today? | no_evidence | no_evidence | pass | This question asks about current law or present applicability. Selected source status: historical. | 8.8 |
| u05 | Which law applies to a murder incident in 2025? | no_evidence | no_evidence | pass | A dated event requires verified applicability and effective-date evidence. The selected source has no such verification. Selected source status: historical. | 9.77 |
| u06 | Will the accused be convicted for an offence in 2024? | no_evidence | no_evidence | pass | A dated event requires verified applicability and effective-date evidence. The selected source has no such verification. Selected source status: historical. | 10.12 |
| u07 | Does the BNSS change procedure for IPC cases? | no_evidence | no_evidence | pass | This question mentions BNS/BNSS/BSA material that is not verified in the selected source. Selected source status: historical. | 13.47 |
| u08 | What does the BSA say about evidence? | no_evidence | no_evidence | pass | This question mentions BNS/BNSS/BSA material that is not verified in the selected source. Selected source status: historical. | 13.74 |

## Actual excerpt responses

The actual chat service was run in provider-free excerpt mode with BM25. Unlike the retrieval table above, the chat service retrieves up to **six** passages, so its hits are not directly comparable to Hit@5. A marker hit means the response included the labelled section on a labelled page. It does not determine whether the passage fully answers the question. Misses and `no_evidence` outputs remain visible.

| ID | Actual mode | Labelled evidence in response | Source pages | Chat ms |
| --- | --- | --- | --- | ---: |
| h01 | excerpts | hit | 55, 55, 1695, 661, 862, 524 | 30.77 |
| h02 | excerpts | MISS | 77, 95, 82, 101, 100, 84 | 21.37 |
| h03 | excerpts | MISS | 86, 103, 322, 325, 321, 86 | 27.32 |
| h04 | excerpts | MISS | 1132, 1024, 1104, 1026, 1016, 1014 | 29.78 |
| h05 | excerpts | MISS | 1026, 1104, 1023, 1013, 1029, 1025 | 28.37 |
| h06 | excerpts | hit | 1095, 1077, 1460, 569, 568, 1095 | 21.94 |
| h07 | excerpts | MISS | 1132, 1014, 1104, 273, 1104, 1025 | 35.68 |
| h08 | excerpts | hit | 1203, 1838, 1199, 1199, 1199, 130 | 27.72 |
| h09 | excerpts | hit | 1201, 1201, 1204, 1202, 1203, 1203 | 32.54 |
| h10 | excerpts | hit | 1839, 1207 | 34.52 |
| h11 | excerpts | hit | 1208, 1215, 1248, 117, 1222, 1199 | 27.67 |
| h12 | excerpts | MISS | 1409, 1413, 1419, 1413, 1419, 1412 | 26.1 |
| h13 | excerpts | MISS | 1413, 1419, 1413, 1412, 1414, 1419 | 29.5 |
| h14 | excerpts | hit | 1850, 1441, 1440, 1441, 1442, 1442 | 19.97 |
| h15 | excerpts | hit | 1503, 1484, 1488, 1488, 1852, 1483 | 21.93 |
| h16 | excerpts | MISS | 1532, 1529, 1544, 1534, 1536, 1534 | 26.85 |
| h17 | excerpts | hit | 1534, 1547, 1542, 1531, 1548, 57 | 24.14 |
| h18 | excerpts | hit | 1588, 1585, 1591, 1587, 1587, 1008 | 29.75 |
| h19 | excerpts | MISS | 1750, 1732, 1733, 1749, 1752, 779 | 18.73 |
| h20 | excerpts | hit | 1765, 1764, 703, 1868, 1765, 1532 | 28.82 |
| h21 | excerpts | MISS | 1836, 1780, 425, 180, 399, 1470 | 24.15 |
| p01 | excerpts | hit | 1409, 1476, 1419, 1407, 1854, 1410 | 33.28 |
| p02 | excerpts | hit | 1199, 1199, 1202, 1244, 1838, 1021 | 27.49 |
| p03 | excerpts | hit | 1764, 731, 130, 1533, 1541, 1764 | 33.16 |
| p04 | excerpts | MISS | 1478, 1488, 1476, 1484, 1476, 1851 | 30.9 |
| p05 | excerpts | hit | 1547, 1431, 1527, 1527, 979, 1531 | 28.74 |
| p06 | excerpts | hit | 55, 53, 53, 1634, 55, 53 | 26.26 |

## Answer-quality limits

Generated answers were **not evaluated**: no generative provider was called. Page-and-section hits and citation membership do not establish that a generated claim is supported or legally correct. The cases are a small development pilot, not a representative or held-out legal benchmark. Ask an independent legal reviewer to annotate answer correctness, evidence support, and appropriate abstention before publishing answer-quality percentages.

## Reproduce

From the repository root, install the locked backend requirements, ensure the configured MiniLM model is cached locally, then run:

```bash
python evaluation/run.py
```

The runner validates the source hash and every labelled section marker. It records requested and actual retrieval methods, returns unsupported-question outcomes, and writes `results.json` and this report. Model or corpus failures are never counted as successful semantic results.
