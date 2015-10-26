#sudokuSolver
-----

## About
When I practice on the leetcode, I happened to find writing a simple sudoku solver is not that difficult. And I happen to know some OpenCV so that I decided to work on a small project that can solve a sudoku by giving a picture of the puzzle.

## Dependecy and usage
All works only verified on OS X. In order to run this program, following packages are needed.

So first you go for

	homebrew

Then use homebrew to get

	opencv
	python
	numpy
	
Now it is good to go to src folder, typing
	
	python solveIt.py /path/to/your/picture

If the puzzle is correctly recognized and there is a solution to the puzzle. A picture with answer will pop out soon. Otherwise it throws an error.


## Credits
OCR is the buildin KNN in OpenCV, while the training sampling I use is from [goo.gl/ZgOfvL](goo.gl/ZgOfvL).  

Fonts are limited and handwritten fonts cannot be handled. Handwritten sample can be found in [MNIST](http://yann.lecun.com/exdb/mnist/). A little bit more work needed to extract bitmap from the binary files.

You can change to your own training samples in
	
	/src/sudoku_parser.py