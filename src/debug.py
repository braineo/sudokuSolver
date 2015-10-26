import sudoku_parser
import sudoku_solver
from cv2 import imshow, waitKey
import
from os import listdir
from os.path import isfile, join
test_path = "../test_picture/"
# sudoku_puzzles = [join(test_path, f) for f in listdir(test_path) if isfile(join(test_path, f))]
sudoku = sudoku_parser.SudokuParser()
solver = sudoku_solver.SudokuSolver()
#for puzzle in sudoku_puzzles:
    #print puzzle
puzzle = sudoku.parse("../test_picture/sudoku9.jpg")
solution = solver.solve(puzzle)
sudoku.draw_solution(solution)
imshow('result', sudoku.resized_largest_square)
waitKey(0)
