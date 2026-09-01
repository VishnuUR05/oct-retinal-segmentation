The dataset consists of five folders

1) "Images": Folder where the original images are downloaded by using the jupyter notebooks inside the Script folder. Images belonging to manual gradings and automatic gradings and also images in original resolution and format are downloaded.
2) "Masks" containting pixel-wise annotations with three manual gradings for 1672 images and 2924 masks with single automatic grading
3) "Boundaries" containing layers as CSV files for individual layers.
4) "Detection" containing CSV files for object detection labels.
and 5) "Scripts" containing jupyterlab notebooks to prepare the data (check README.txt inside).

The dataset covers semantic segmentation and object detection tasks.
For semantic segmentation the layers/bands included are ILM, OPL-Henles, IS/OS junction, IBRPE and OBRPE.
For object detection the labels cover choroidal folds, fluid, geographicatrophy, harddrusen, hyperfluorescentspots, PR layerdisruption, reticular drusen, softdrusen, and softdrusen PED.

The tree structure after preparation looks as following:

├── Images
│   ├── Images_Automatic
│   │   ├── AMD Part1
│   │   ├── AMD Part2
│   │   ├── DME
│   │   ├── DRUSEN
│   │   ├── Normal Part1
│   │   └── Normal Part2
│   ├── Images_Manual
│   │   ├── AMD Part1
│   │   ├── AMD Part2
│   │   ├── DME
│   │   ├── Normal Part1
│   │   └── Normal Part2
│   └── Images_Original
│       ├── AMD Part1
│       ├── AMD Part2
│       ├── DME
│       ├── DRUSEN
│       ├── Normal Part1
│       └── Normal Part2
├── Masks
│   ├── Masks_Automatic
│   │   └── Grading
│   │       ├── AMD Part1
│   │       ├── AMD Part2
│   │       ├── DME
│   │       ├── DRUSEN
│   │       ├── Normal Part1
│   │       └── Normal Part2
│   ├── Masks_Automatic_RGB
│   │   └── Grading
│   │       ├── AMD Part1
│   │       ├── AMD Part2
│   │       ├── DME
│   │       ├── DRUSEN
│   │       ├── Normal Part1
│   │       └── Normal Part2
│   ├── Masks_Manual
│   │   ├── Grading_1
│   │   │   ├── AMD Part1
│   │   │   ├── AMD Part2
│   │   │   ├── DME
│   │   │   ├── Normal Part1
│   │   │   └── Normal Part2
│   │   ├── Grading_2
│   │   │   ├── AMD Part1
│   │   │   ├── AMD Part2
│   │   │   ├── DME
│   │   │   ├── Normal Part1
│   │   │   └── Normal Part2
│   │   └── Grading_3
│   │       ├── AMD Part1
│   │       ├── AMD Part2
│   │       ├── DME
│   │       ├── Normal Part1
│   │       └── Normal Part2
├── └── Masks_RGB
│       ├── Grading_1
│       │   ├── AMD Part1
│       │   ├── AMD Part2
│       │   ├── DME
│       │   ├── Normal Part1
│       │   └── Normal Part2
│       ├── Grading_2
│       │   ├── AMD Part1
│       │   ├── AMD Part2
│       │   ├── DME
│       │   ├── Normal Part1
│       │   └── Normal Part2
│       └── Grading_3
│           ├── AMD Part1
│           ├── AMD Part2
│           ├── DME
│           ├── Normal Part1
│           └── Normal Part2
├── Boundaries
│   ├── Boundaries_Automatic
│   │   └── Grading
│   │       └── Output
│   └── Boundaries_Manual
│       ├── Grading_1
│       │   ├── AMD Part1
│       │   ├── AMD Part2
│       │   ├── DME
│       │   ├── Normal Part1
│       │   └── Normal Part2
│       ├── Grading_2
│       │   ├── AMD Part1
│       │   ├── AMD Part2
│       │   ├── DME
│       │   ├── Normal Part1
│       │   └── Normal Part2
│       └── Grading_3
│           ├── AMD Part1
│           ├── AMD Part2
│           ├── DME
│           ├── Normal Part1
│           └── Normal Part2
├── Detection
│   └── Images
│       ├── AMD Part1
│       ├── AMD Part2
│       └── DRUSEN
