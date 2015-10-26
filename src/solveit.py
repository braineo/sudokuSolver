import sudoku_parser
import sudoku_solver
from cv2 import imshow, waitKey
from sys import argv

sudoku = sudoku_parser.SudokuParser()
solver = sudoku_solver.SudokuSolver()
if not argv[1]:
    raise IndexError
puzzle = sudoku.parse(argv[1])
solution = solver.solve(puzzle)
sudoku.draw_solution(solution)
imshow('result', sudoku.resized_largest_square)
waitKey(0)
