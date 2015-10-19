__author__ = 'braineo'

import sudoku_parser
import sudoku_solver
sudoku = sudoku_parser.SudokuParser()
solver = sudoku_solver.SudokuSolver()
puzzle = sudoku.parse("../test_picture/sudoku1.jpg")
solution = solver.solve(puzzle)
print solution