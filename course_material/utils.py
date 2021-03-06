import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
    
def test_setup():
    import numpy as np
    import scipy as sp
    import matplotlib as plt
    import sys
    
    print(f'Python Version : \t{sys.version.split(" | ")[0]}')
    print(f'OpenCV Version : \t{cv2.__version__}')
    print(f'Numpy Version :  \t{np.__version__}')
    print(f'SciPy Version :  \t{sp.__version__}')
    print(f'Matplotlib Version : \t{plt.__version__}')
    
def imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
