"""
Logic module to solve a Sudoku puzzle.
Using basic backtracking to find out a valid solution.
Code bring from my LeetCode practice, maybe will update the algorithm when I got how Algorithm X works
"""
import copy

NUM_ROWS = 9
NUM_BLOCKS = 3

class SudokuSolver(object):

    """
    Solves a Sudoku puzzle.

    """
    def __init__(self):
        """
        initialize Sudoku solver, alloc for judgement bits
        """
        self.row = [[0] * 10 for _ in xrange(NUM_ROWS)]
        self.col = copy.deepcopy(self.row)
        self.block = copy.deepcopy(self.row)

    def solve(self, puzzle):
        """
        Solve a Sudoku puzzle
        :param puzzle: numpy.ndarray 2-D array of a Sudoku puzzle
        :return solution: numpy.ndarray 2-D array of solution for the Sudoku puzzle
        """
        # initiate judgment flags according to puzzle
        for i in xrange(NUM_ROWS):
            for j in xrange(NUM_ROWS):
                if puzzle[i, j] != 0:
                    d = puzzle[i, j]
                    self.row[i][d] = \
                        self.col[j][d] = \
                        self.block[i//NUM_BLOCKS*NUM_BLOCKS+j//NUM_BLOCKS][d] = 1
        solution = copy.deepcopy(puzzle)
        # search from top left if there is a solution for the Sudoku puzzle
        have_solution = self._search_solution(solution, 0, 0)
        if have_solution:
            return solution
        else:
            raise ContradictionError

    def _search_solution(self, solution, i, j):
        """
        Search available solution for the Sudoku puzzle
        :param solution: numpy.ndarray solution for the Sodoku puzzle
        :param i: int x axis index of digit on the puzzle
        :param j: int y axis index of digit on the puzzle
        :return: Boolean if the puzzle have a solution
        """
        # Return True if location is at bottom right
        if i*9+j >= 81:
            return True
        nexti, nextj = (i+1, 0) if j>=NUM_ROWS-1 else (i, j+1)
        if solution[i, j] != 0:
            return self._search_solution(solution, nexti, nextj)
        else:
            block_num = i//NUM_BLOCKS*NUM_BLOCKS+j//NUM_BLOCKS
            for d in xrange(1, 10):
                if self.row[i][d] or\
                    self.col[j][d] or \
                    self.block[block_num][d]:
                    continue
                else:
                    self.row[i][d] = self.col[j][d] = \
                    self.block[block_num][d] = 1
                    solution[i, j] = d
                    if self._search_solution(solution, nexti, nextj):
                        return True
                    self.row[i][d] = self.col[j][d] = \
                    self.block[block_num][d] = 0
                    solution[i, j] = 0
            return False


class ContradictionError(Exception):
    """
    Contradition found in the puzzle
    """
