# template matching - Immunogold Particle Labeling

import cv2
import numpy as np

template_paths = ['assets/Gold_Particle1.PNG', 'assets/Gold_Particle_small.PNG']

images = ('assets/S22 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S22_0003.tif',
          'assets/S29 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S29.tif', 'assets/S15 MBTt FFRIL01 R1Bg1d Wt 8wk AMPA6nm_NR1_12nm_vGlut2_18nm S15.tif')

# Load and Resize Main Image
for image in images:
    img = cv2.imread(image, 0)
    img = cv2.resize(img, (600, 600))
    img2 = img.copy()

    # Define Method Used (trial and error)
    method = cv2.TM_CCOEFF_NORMED

    # Perform Convolution (template)
    for template_path in template_paths:
        template = cv2.imread(template_path, 0)
        h, w = template.shape
        result = cv2.matchTemplate(img2, template, method)
        threshold = 0.6  # Adjust as needed
        locations = np.where(result >= threshold)

        # Draws Rectangle
        for pt in zip(*locations[::-1]):
            bottom_right = (pt[0] + w, pt[1] + h)
            cv2.rectangle(img2, pt, bottom_right, 255, 2)

    # Show Final Image
    cv2.imshow('Match Image', img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
