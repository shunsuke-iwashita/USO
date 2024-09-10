# USO (Ultimate Scoring Opportunity)

This repository contains the implementation of “Space evaluation based on pitch control using drone video in Ultimate” to be presented at CASSIS. [[website](http://www.cascadiasports.com/)] The code supports the methods and experiments presented in the paper. [[ReserchGate](https://www.researchgate.net/publication/383879446_Space_evaluation_based_on_pitch_control_using_drone_video_in_Ultimate)]

## Video

https://drive.google.com/drive/folders/13yr0bzPIdAkum4YztNkj-nxba5r5ohO-?usp=sharing


## Overview

This repository contains the following:

- Data for demonstration
- Demonstration Results
- Code to calculate OBSO and USO

## Getting Started

### Installation

1. Clone this repository:

```bash
$ git clone https://github.com/shunsuke-iwashita/USO.git
```

2. Create environment:

```bash
$ pip install -r requirements.txt
```

### Perform Demonstrations

1. Calculate and visualize OBSO

```bash
$ python calculate_obso.py --id 1_1_00
```

2. Calculate and visualize USO

```bash
$ python calculate_uso.py
```

## Contact

If you have any questions, please contact author:

- Shunsuke Iwashita (iwashita.shunsuke[at]g.sp.m.is.nagoya-u.ac.jp)
