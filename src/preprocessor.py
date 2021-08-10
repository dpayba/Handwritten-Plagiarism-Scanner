import cv2
import numpy as np
import random
from typing import Tuple

from load_data import Batch

class Preprocessor:
    def __init__(self, img_size: Tuple[int, int],
                 padding: int=0, width_dynamic: bool=False,
                 data_augmentation: bool=False, line_mode: bool=False):
        assert not (width_dynamic and data_augmentation)
        assert not (padding > 0 and not width_dynamic)

        self.image_size = img_size
        self.padding = padding
        self.width_dynamic = width_dynamic
        self.data_augmentation = data_augmentation
        self.line_mode = line_mode

    def simulate_text_line(self, batch):
        # create text line by combining word images

        default_word_separation = 30
        default_n_words = 5

        # iterate batch
        result_images = []
        res_get_texts = []
        for i in range(batch.batch_size):
            # number words to put on line
            n_words = random.randint(1, 8) if self.data_augmentation else default_n_words

            # concatenate ground truth
            current_ground_truth = ' '.join([batch.gt_texts[(i + j) % batch.batch_size] for j in range(n_words)])
            res_get_texts.append(current_ground_truth)

            # add word images to list, compute image size
            selected_images = []
            word_separation = [0]
            height = 0
            width = 0
            for j in range(n_words):
                current_selected_img = batch.imgs[(i + j) % batch.batch_size]
                current_word_separation = random.randint(20, 50) if self.data_augmentation else default_word_separation
                height = max(height, current_selected_img.shape[0])
                width += current_selected_img.shape[1]
                selected_images.append(current_selected_img)
                if j + 1 < n_words:
                    width += current_word_separation
                    word_separation.append(current_word_separation)

            # collect images into target
            target_image = np.ones([height, width], np.uint8) * 255
            x = 0
            for current_selected_img, current_word_separation in zip(selected_images, word_separation):
                x += current_word_separation
                y = (height - current_selected_img.shape[0]) // 2
                target_image[y:y + current_selected_img.shape[0]:, x:x + current_selected_img.shape[1]] = current_selected_img
                x += current_selected_img.shape[1]

            result_images.append(target_image)
        return Batch(result_images, res_get_texts, batch.batch_size)

    def process_image(self, img: np.ndarray):
        if img is None:
            img = np.zeros(self.image_size[::-1])

        # convert image type
        img = img.astype(np.float)

        if self.data_augmentation:
            if random.random() < 0.25:
                def random_odd():
                    return random.randint(1, 3) * 2 + 1
                img = cv2.GaussianBlur(img, (random_odd(), random_odd()), 0)
            if random.random() < 0.25:
                img = cv2.dilate(img, np.ones((3, 3)))
            if random.random() < 0.25:
                img = cv2.erode(img, np.ones((3, 3)))

            # augment data geometrically
            wt, ht = self.image_size
            height, width = img.shape
            func = min(wt / width, ht / height)
            f_x = func * np.random.uniform(0.75, 1.05)
            f_y = func * np.random.uniform(0.75, 1.05)

            tx_center = (wt - width * f_x) / 2
            ty_center = (ht - height * f_y) / 2
            freedom_x = max((wt - f_x * width) / 2, 0)
            freedom_y = max((ht - f_y * height) / 2, 0)
            tx = tx_center + np.random.uniform(-freedom_x, freedom_x)
            ty = ty_center + np.random.uniform(-freedom_y, freedom_y)

            # map image to target
            M = np.float32([[f_x, 0, tx], [0, f_y, ty]])
            target = np.ones(self.image_size[::-1]) * 255
            img = cv2.warpAffine(img, M, dsize=self.image_size, dst=target, borderMode=cv2.BORDER_TRANSPARENT)

            if random.random() < 0.5:
                img = img * (0.25 + random.random() * 0.75)
            if random.random() < 0.25:
                img = np.clip(img + (np.random.random(img.shape) - 0.5) * random.randint(1, 25), 0, 255)
            if random.random() < 0.1:
                img = 255 - img

        else:
            if self.width_dynamic:
                ht = self.image_size[1]
                h, w = img.shape
                f = ht / h
                wt = int(f * w + self.padding)
                wt = wt + (4 - wt) % 4
                tx = (wt - w * f) / 2
                ty = 0
            else:
                wt, ht = self.image_size
                h, w = img.shape
                f = min(wt / w, ht / h)
                tx = (wt - w * f) / 2
                ty = (ht - h * f) / 2

            # map image into target image
            M = np.float32([[f, 0, tx], [0, f, ty]])
            target = np.ones([ht, wt]) * 255
            img = cv2.warpAffine(img, M, dsize=(wt, ht), dst=target, borderMode=cv2.BORDER_TRANSPARENT)

            # transpose for TF
        img = cv2.transpose(img)

        # convert to range [-1, 1]
        img = img / 255 - 0.5
        return img

    @staticmethod
    def truncate_label(text, max_text_length):
        cost = 0
        for i in range(len(text)):
            if i != 0 and text[i] == text[i-1]:
                cost += 2
            else:
                cost += 1
            if cost > max_text_length:
                return text[:i]
        return text

    def process_batch(self, batch):
        if self.line_mode:
            batch = self.simulate_text_line(batch)

        res_images = [self.process_image(img) for img in batch.imgs]
        max_text_len = res_images[0].shape[0] // 4
        res_gt_texts = [self.truncate_label(gt_text, max_text_len) for gt_text in batch.generated_texts]
        return Batch(res_images, res_gt_texts, batch.batch_size)

def main():
    import matplotlib.pyplot as plt

    img = cv2.imread('../data/test.png', cv2.IMREAD_GRAYSCALE)
    img_augmented = Preprocessor((256, 32), data_augmentation=True).process_image(img)
    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(cv2.transpose(img_augmented) + 0.5, 'gray', 0, 1)
    plt.show()

if __name__ == '__main__':
    main()