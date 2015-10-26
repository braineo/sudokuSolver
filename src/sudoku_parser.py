""" Recognize sudoku image and parse sudoku puzzle via OCR
OCR is implemented by KNN
"""

import cv2
import numpy as np

SUDOKU_RESIZE = 450  # Pixel size of resized sudoku puzzle
NUM_ROWS = 9
OFFSET = SUDOKU_RESIZE/NUM_ROWS//2
SAMPLE = '../training_data/feature_vector_pixels2.data'
LABEL = '../training_data/samples_pixels2.data'

class SudokuParser(object):

    """ Parse sudoku puzzle

    Attributes:
    image: numpy.ndarray of origin Sudoku image
    resize_largest_square: numpy.ndarray of the largest square in the image
    model: cv2.KNearest of K-Nearest Neighbors classifier using training data
    puzzle: string of puzzle number
    """

    def __init__(self):
        """Initialize class and KNN model
        :return: None
        """

        self.model = self._get_model()

    def _get_model(self):
        """
        :return model: K-Nearest Neighbors classifier using training data
        :rtype: cv2.KNearest
        """

        training_sample = np.float32(
            np.loadtxt(SAMPLE))
        training_response = np.float32(
            np.loadtxt(LABEL))

        model = cv2.KNearest()
        model.train(training_sample, training_response)
        return model

    def parse(self, image_data):
        """
        Parse Sudoku image file into [ str[] ]
        :param image_data: string image path
        :return self.puzzle: String numbers in the Sudoku puzzle
        """

        self.image = cv2.imread(image_data)
        square = self._find_largest_square()
        self.resized_largest_square = self._perspective_sudoku_image(
            square, SUDOKU_RESIZE)
        self.puzzle = self._get_puzzle()
        return self.puzzle

    def draw_solution(self, solution):
        """
        Draw solution of the puzzle on the iamge
        :param solution: numpy.ndarray solution to Sudoku puzzle
        :return: numpy.ndarray picture with solution on it
        """

        for i, row in enumerate(solution):
            cell_size = SUDOKU_RESIZE // NUM_ROWS
            celli = i * cell_size
            for j, d in enumerate(row):
                if self.puzzle[i, j] == 0:
                    cellj = j * cell_size
                    cv2.putText(self.resized_largest_square,
                                str(int(d)),
                                (cellj+OFFSET, celli+OFFSET),
                                cv2.FONT_HERSHEY_PLAIN,
                                fontScale=1.2,
                                color=(100, 100, 0),
                                thickness=2)
        return self.resized_largest_square

    def _find_largest_square(self):
        """
        Find contour of the largest square in the image
        :return:
        """

        contours, image = self._get_major_contours(
            self.image, threshold_type=cv2.THRESH_BINARY_INV)
        possible_puzzles = {}
        for contour in contours:
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, closed=True)
            # Approximate contour into a polygon so that it can be judged
            contour = cv2.approxPolyDP(contour, length * 0.02, closed=True)
            # if contour has 4 vertices and it is convex, it is possible a
            # puzzle
            if len(contour) == 4 and (
                    cv2.isContourConvex(contour)):
                possible_puzzles[area] = contour

        areas = possible_puzzles.keys()
        areas.sort()
        return possible_puzzles[areas[-1]]

    def _get_puzzle(self):
        """
        Retrieve sudoku matrix
        :return: [[int]]
        """

        sudoku_matrix = np.zeros((NUM_ROWS, NUM_ROWS), np.uint8)

        contours, image = self._get_major_contours(self.resized_largest_square,
                                                   sigma1=0,
                                                   dilate=False,
                                                   threshold_type=cv2.THRESH_BINARY_INV)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        erode = cv2.erode(image, kernel)
        dilate = cv2.dilate(erode, kernel)

        #cv2.imshow("largest square", erode)
        # cv2.waitKey()
        #dilate = erode
        image_copy = dilate.copy()
        contours, hierarchy = cv2.findContours(
            image_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # if 100 < area < 800:
            if 50 < area < 800:
                (bx, by, bw, bh) = cv2.boundingRect(contour)
                # if (100 < bw*bh < 1200) and (10 < bw < 40) and (25 < bh < 45):
                # aju
                if (100 < bw * bh < 1200) and (5 < bw < 40) and (10 < bh < 45):
                    # Get the region of interest, which contains the number.
                    roi = dilate[by:by + bh, bx:bx + bw]
                    small_roi = cv2.resize(roi, (10, 10))
                    #cv2.imshow('tmp', roi)
                    # cv2.waitKey()
                    feature = small_roi.reshape((1, 100)).astype(np.float32)

                    # Use the model to find the most likely number.
                    ret, results, neigh, dist = self.model.find_nearest(
                        feature, k=1)
                    integer = int(results.ravel()[0])

                    # gridx and gridy are indices of row and column in Sudoku
                    gridy = (bx + bw / 2) / (SUDOKU_RESIZE / NUM_ROWS)
                    gridx = (by + bh / 2) / (SUDOKU_RESIZE / NUM_ROWS)
                    sudoku_matrix.itemset((gridx, gridy), integer)

        print sudoku_matrix
        return sudoku_matrix

    def _process_single_digit_image(self, image):
        pass

    def _get_major_contours(self, image, sigma1=0, dilate=True,
                            threshold_type=cv2.THRESH_BINARY):
        """
        Preprocess the image (turn into gray scale and guassian blur) and get major contours
        :param image: numpy.ndarray sukudo image
        :param sigma1: Integer Gaussian kernel standard deviation in X direction
        :param dilate: Boolean apply dilate to image
        :param threshold_type: Integer threshold type
        :return: [] list of contours, numpy.ndarray preprocessed image
        """
        #resize_factor = float(SUDOKU_RESIZE) / image.shape[0]
        #image = cv2.resize(image, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_NEAREST)
        try:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        except cv2.error as e:
            raise ImageError('Could not process image.')

        blur_image = cv2.GaussianBlur(gray_image, (7, 7), sigma1)
        thresh = cv2.adaptiveThreshold(blur_image,
                                       maxValue=255,
                                       adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                       thresholdType=threshold_type,
                                       blockSize=25,
                                       C=10)

        if dilate:
            thresh = cv2.dilate(thresh,
                                kernel=cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=(3, 3)))

        return_image = thresh.copy()
        contours, hierachy = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, return_image

    def _reorder_vertices(self, square):
        """
        Reorder vertices in clockwise order
        :param square: numpy.ndarray of vertices
        :return square_reordered: numpy.ndarray of reordered square vertices
        """
        square = square.reshape((4, 2))
        square_reordered = np.zeros((4, 2), dtype=np.float32)
        # top-left smallest coordinate sum
        # bottom-right largest coordinate sum
        sum = np.sum(square, axis=1)
        square_reordered[0] = square[np.argmin(sum)]
        square_reordered[2] = square[np.argmax(sum)]

        # top-right smallest coordinate difference
        # bottom-left largest coordinate difference
        diff = np.diff(square, axis=1)
        square_reordered[1] = square[np.argmin(diff)]
        square_reordered[3] = square[np.argmax(diff)]

        return square_reordered

    def _perspective_sudoku_image(self, square, size):
        """
        Resize image(numpy.ndarray) and perform perspective transform before getting digits on sudoku
        :param square: numpy.ndarray 4 vertices on image
        :param size: int size of one edge
        :return transformed_image: squared sudoku image
        """
        reordered_square = self._reorder_vertices(square)
        # target vertices in clockwise order
        target_vertices = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)

        transform_matrix = cv2.getPerspectiveTransform(
            reordered_square, target_vertices)
        transformed_image = cv2.warpPerspective(
            self.image, transform_matrix, (size, size))

        return transformed_image


class ImageError(Exception):

    """ Raised when image could not be processed.
    """
