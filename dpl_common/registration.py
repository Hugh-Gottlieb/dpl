import cv2
import numpy as np

from dpl_common.helpers import Image

class Registration:

    def __init__(self):
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

    def register_image(self, image: Image, target: Image):
        image.data = self.register_image(image.data, target.data)

    def register_images(self, images: list[Image], target: Image):
        for image in images:
            image.data = self.register_image(image.data, target.data)

    def register_imgs(self, imgs: list[np.ndarray], target: np.ndarray) -> list[np.ndarray]:
        return [self.register_img(img, target) for img in imgs]

    def register_img(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype("uint8")
        target_norm = cv2.normalize(target, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype("uint8")
        keypoints1, descriptors1 = self.feature_detector.detectAndCompute(img_norm, None)
        keypoints2, descriptors2 = self.feature_detector.detectAndCompute(target_norm, None)
        matches = list(self.matcher.match(descriptors1, descriptors2, None))
        matches.sort(key=lambda x: x.distance, reverse=False)
        # NOTE - could sort through matches and remove low scoring, and /or those that move too much
        assert (len(matches) >= 4), "Insufficient matches found during registration"
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        homography, _ = cv2.findHomography(points2, points1, cv2.RANSAC)
        height, width = img.shape
        return cv2.warpPerspective(img, homography, (width, height), borderValue=0)