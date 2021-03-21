# Multimodal Humor Detection

## Introduce

This project is my own Undergraduate Graduation Project.

It implements the C-MFN baseline proposed in [UR-FUNNY](https://www.aclweb.org/anthology/D19-1211) to detect humor in multimodal language.

All the code of the project is in the main.py program.

## Experiment Result

### accuracy

The data in the table are the average of 10 experimental results.

| Modality |  T   | A+V  | T+A  | T+V  |  T+A+V  |
| :------: | :--: | :--: | :--: | :--: | :-----: |
|  C-MFN   |      |      |      |      | 0.63236 |
| C-MFN(P) |      |      |      |      |         |
| C-MFN(C) |      |      |      |      |         |

### train cost

The data in the table are the average of 10 experimental results.

All programs run on **Nvidia Tesla T4**.

| Modality |  T   | A+V  | T+A  | T+V  |        T+A+V        |
| :------: | :--: | :--: | :--: | :--: | :-----------------: |
|  C-MFN   |      |      |      |      | 240.55228493213653s |
|          |      |      |      |      |                     |
|          |      |      |      |      |                     |

