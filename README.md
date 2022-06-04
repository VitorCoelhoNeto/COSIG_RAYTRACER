# **COSIG_RAYTRACER**
**Raytracer** developed in **Python 3.10.2**

## **Installation**

Use the package manager **[pip](https://pip.pypa.io/en/stable/)** to install the required libraries.

Contained in requirements.txt, these are:
1. PyQt5 -> version 5.15.6, used for the UI
2. pyqt5-tools -> version ??, used for the Qt Designer
3. numpy -> version 1.22.3, used for mathematical operations

```bash
pip install requirements.txt
```

## **Version 0.1**
In this installation, some classes have been developed for future use on the Raytracer, these include some scene objects, triangles, sphere, etc. and other raytracing specific objects such as Ray, Hit, etc. 

In this version:
```bash
python main.py
```
parses the sample scene file contained in the relative directory:
``` Python
FILE_PATH = "..\test_scene_1.txt"
```

Used **Python** native library imports:
```Python
import json
import sys
import re
```
Used **Python** PyQt5 library imports:
```Python
from PyQt5 import QtCore, QtGui, QtWidgets
```

Used **Python** numpy library imports:
```Python
import numpy as np
```

This version's function's documentation follows [Python's recommended documentation guidelines](https://realpython.com/documenting-python-code/), where in **:param *type* *var*: *description***:

* **1. *param*** indicates one of the function's parameters that needs to be passed;

* **2. *type*** indicates the parameter's type (**e.g.** *str*, *int*, *etc*.);

* **3. *var*** is the parameters name;

* **4. *description*** is the parameters description.

in **:returns: *var***:

* **1. *returns*** indicates something is returned in that function;
* **2. *var*** is the returned object.

and in **:rtype: *type***:

* **1. *rtype*** indicates that the returned object's type is going to be indicated;
* **2. *type*** is the returned object's type (**e.g.** *str*, *int*, *etc*.).

## License
**[MIT](https://choosealicense.com/licenses/mit/)**
