import cv2
import numpy as np

from dpl_common.helpers import Image

class Registration:

    def __init__(self, feature_limit:int = None):
        self.feature_detector = cv2.SIFT_create(nfeatures=feature_limit)
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_L1)

    def register_image(self, image: Image, target: Image):
        image.data = self.register_img(image.data, target.data)

    def register_images(self, images: list[Image], target: Image, skip_target=False):
        for image in images:
            if skip_target and image == target:
                continue
            image.data = self.register_img(image.data, target.data)

    def register_imgs(self, imgs: list[np.ndarray], target: np.ndarray) -> list[np.ndarray]:
        return [self.register_img(img, target) for img in imgs]

    # NOTE: currently recalculate the target keypoints / descriptors for every image, could speed up by reusing
    # NOTE: input is typically a uint16, but return a float64 which nan's at the border
    def register_img(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        target_median = cv2.medianBlur(target, 5)
        img_median = cv2.medianBlur(img, 5)
        baseline = np.min(target_median)
        ratio = (np.max(target_median) - baseline) / 255
        target_norm = ((target_median - baseline) / ratio).astype(np.uint8)
        img_norm = np.clip(((img_median - baseline) / ratio), 0, 255).astype(np.uint8)
        keypoints1, descriptors1 = self.feature_detector.detectAndCompute(target_norm, None)
        keypoints2, descriptors2 = self.feature_detector.detectAndCompute(img_norm, None)
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
        return cv2.warpPerspective(img.astype(float), homography, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)