import numpy as np
# from sudoku_generator import generate_sudoko
import itertools
import time

n_num = 9

# print("Input the desired difficulty(easy, medium, hard, extreme)")
# level = input()
# board = generate_sudoko(level)



class Cell():
    def __init__(self,row, col, num=None):
        self.row = row
        self.col = col
        self.num = 0
        self.possible_nums = set(np.arange(1,n_num+1))
        self.solved = False

        # If it gets a num at start it's already solved
        if num is not None:
            self.possible_nums = {num}
            self.solved = True
            self.num = num


    # Returns true of something changes
    def update_possible_num(self, num):
        if self.solved:
            return False

        original_length = len(self.possible_nums)
        self.possible_nums.difference_update(num)
        new_length = len(self.possible_nums)

        if new_length <= 0:
            raise ValueError("No possible value in cell. Invalid sodoku.")
        return not original_length == new_length

    def check_if_only_one_possible_num(self):
        if len(self.possible_nums) == 1:
            self.num = list(self.possible_nums)[0]
            self.solved = True

    def __str__(self):
        return "Row: " + str(self.row) + ". Col: " + str(self.col) + ". Num: " + str(self.num) + ". Possible nums: " + str(self.possible_nums) 

def cell_board(board):
    board_ = []
    for row_index,row in enumerate(board):
        for col,num_ in enumerate(row):
            num_ = num_ if not num_==0 else None 
            cell = Cell(row_index, col, num = num_)
            board_.append(cell)
    
    board_ = np.array(board_, dtype='object')
    board_ = board_.reshape(board.shape)
    return board_


def get_num_from_cells(cells):
    # Handle case where it's only 1 element
    if cells.size == 1:
        return cells[0].num

    # Handle the case where cells is 1D
    row_vector = False
    col_vector = False
    x = 1
    y = 1
    if len(cells.shape) == 1:
        # If it's a row vector:
        if cells[0].row == cells[1].row:
            row_vector = True
            x = cells.shape[0]
        else:
            col_vector = True
            y = cells.shape[0]
    else:
        y,x = cells.shape

    nums = np.zeros(cells.shape, dtype='int')
    
    for i in range(cells.size):
        cell = cells.reshape(-1)[i]

        if row_vector:
            col = i
            nums[col] = cell.num
        elif col_vector:
            row = i
            nums[row] = cell.num
        else:
            row = (i // x)
            col = (i % x)
            nums[row, col] = cell.num
    
    return nums.reshape(-1)

def cells_print(cells):
    nums = get_num_from_cells(cells)
    nums = nums.reshape(cells.shape)
    
    print(nums)

def board_print(board):
    # Stolen from https://stackoverflow.com/questions/45471152/how-to-create-a-sudoku-puzzle-in-python
    base = int(n_num**0.5) # No fucking idea.
    side = n_num
    def expandLine(line):
        return line[0]+line[5:9].join([line[1:5]*(base-1)]*base)+line[9:13]
    line0  = expandLine("╔═══╤═══╦═══╗")
    line1  = expandLine("║ . │ . ║ . ║")
    line2  = expandLine("╟───┼───╫───╢")
    line3  = expandLine("╠═══╪═══╬═══╣")
    line4  = expandLine("╚═══╧═══╩═══╝")

    symbol = " 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    nums   = [ [""]+[symbol[n.num] for n in row] for row in board ]
    print(line0)
    for r in range(1,side+1):
        print( "".join(n+s for n,s in zip(nums[r-1],line1.split("."))) )
        print([line2,line3,line4][(r%side==0)+(r%base==0)])

# Remove nums by logic exclusion
def exclusion_cells(cells):
    possible_nums = []

    cells_as_row = cells.reshape(-1)
    for cell in cells_as_row:
        possible_nums.append(cell.possible_nums)
    possible_nums = np.array(possible_nums)

    # Check if multiple sets are the same
    n_copies = np.zeros((cells.size,), dtype='int')
    for i, set_ in enumerate(possible_nums):
        for set__ in possible_nums:
           n_copies[i] += set_==set__ 

    # If there are a possible num set with n elements there needs to be n cells with that possible num sets for anything to be excluded
    good_possible_nums = []
    for i in range(0,cells.size):
        if len(possible_nums[i]) == n_copies[i]:
            good_possible_nums.append(i)
    
    # Remove those possible nums from all cells which don't have exactly them as possible nums
    changed_something = False
    for i in good_possible_nums:
        update_set = possible_nums[i]
        for cell in cells:
            if not cell.possible_nums == update_set:
                changed_something |= cell.update_possible_num(update_set)

    return changed_something

def exclusion_cells_row(row):
    return exclusion_cells(row)
    
def exclusion_cells_col(col):
    return exclusion_cells(col)  

def exclusion_cells_square(square):
    return exclusion_cells(square.reshape(-1))

# For inputting the board into a sudoko solver website
def print_seed(board):
    output = ""
    for cell in board.reshape(-1):
        if cell.num == 0:
            output += " "
        else:
            output += str(cell.num)
    print(output)

# Solve hidden numbers, https://www.sudoku-solutions.com/index.php?section=solvingHiddenSubsets#hiddenSingle
def find_hidden_num_cells(cells):
    cells = cells.reshape(-1)
    
    # Find n possible nums which are unique to n cells
    cells_with_unique_possible_num = []
    checked_sets = []
    for cell in cells:
        possible_sets = [set(itertools.combinations(cell.possible_nums,i)) for i in range(1,len(list(cell.possible_nums))+1)]

        # Convert the touple in possible_sets to sets
        possible_sets_ = []
        for set_ in possible_sets:
            for set__ in set_:
                if type(set__) == tuple:
                    possible_sets_.append(set(list(set__)))
                else:
                    possible_sets_.append({set__})
        
        possible_sets = possible_sets_
        
        for set_ in possible_sets:
            if not set_ in checked_sets:
                cells_with_sets = []
                checked_sets.append(set_)
                for cell_ in cells:
                    if set_.issubset(cell_.possible_nums):
                        cells_with_sets.append(cell_)
                
                if len(cells_with_sets) == len(list(set_)):
                    cells_with_unique_possible_num.append([set_, cells_with_sets])

    
    # Make sure non of the elements in the unique sets is in any other set
    cells_with_unique_possible_num_ = []
    for list_ in cells_with_unique_possible_num:
        nums = list_[0]
        cells_ = list_[1]
        
        element_possible_in_other_cell = False
        for cell in cells:
            if not cell in cells_:
                if not 0 == len(nums.intersection(cell.possible_nums)):
                    element_possible_in_other_cell = True

        if not element_possible_in_other_cell:
            cells_with_unique_possible_num_.append(list_)

    cells_with_unique_possible_num = cells_with_unique_possible_num_


    # Remove all the non unique possible nums from the cells with unique possible nums
    update_something = False
    for list_ in cells_with_unique_possible_num:
        nums = list_[0]
        cells_ = list_[1]
        for cell in cells_:
            update_set = set([i for i in range(1,n_num+1)]) - nums
            update_something |= cell.update_possible_num(update_set)
    
    return update_something

def find_hidden_num_square(square):
    return find_hidden_num_cells(square)

def find_hidden_num_row(row):
    return find_hidden_num_cells(row)

def find_hidden_num_col(col):
    return find_hidden_num_cells(col)
    
#Find pointing nummbers : https://www.sudoku-solutions.com/index.php?section=solvingInteractions#pointingPair
def pointing_numbers_cells(board, cells, cell_type):
    cells = cells.reshape(-1)

    # Find all unique possible nums inside the given cells
    possible_nums = []
    for cell in cells:
        possible_sets = [set(itertools.combinations(cell.possible_nums,i)) for i in range(1,len(list(cell.possible_nums))+1)]
        
        # Save all the possible nums
        for set_ in possible_sets:
            for set__ in set_:
                if not set__ in possible_nums:
                    possible_nums.append(set(set__))

    # Make sure the possible nums found are actually missing from the cells
    cell_nums = get_num_from_cells(cells)
    tmp_list = []
    for possible_num in possible_nums:
        needed = True
        for element in list(possible_num):
            if element in cell_nums:
                needed = False
                break
        
        if needed and not possible_num in tmp_list:
            tmp_list.append(possible_num)

    possible_nums = tmp_list

    # Group them into lists with the same set of possible nums
    pairs = []
    for possible_num in possible_nums:
        tmp_list = []
        for cell in cells:
            #if possible_num.issubset():
            if possible_num.issubset(cell.possible_nums):
                tmp_list.append(cell)
        pairs.append((possible_num, tmp_list))

    # Make sure the possible nums for the cells cannot be in any other cell in cells
    tmp_list = []
    for tuple_ in pairs:
        (possible_num, pair) = tuple_
        can_be_in_other_cell = False
        for cell in cells:
            if not cell in pair:
                for element in list(possible_num):
                    if element in cell.possible_nums:
                        can_be_in_other_cell = True
                        break
        
        if not can_be_in_other_cell:
            tmp_list.append(tuple_)

    pairs = tmp_list
       
    def in_same_square(cells):
        same_square_col = any([all([cell.row < i*3 and cell.row >= (i-1)*3 for cell in cells]) for i in range(1,4)])
        same_square_row = any([all([cell.col < i*3 and cell.col >= (i-1)*3 for cell in cells]) for i in range(1,4)])
        return same_square_col and same_square_row
    
    # Only if the matching pairs are on the same row, col or square can they be used. Extract those who are
    row_pairs = []
    col_pairs = []
    square_pairs = []
    for tuple_ in pairs:
        (possible_num, pair) = tuple_
        row = pair[0].row
        col = pair[0].col 
        
        # Check if the are row/col pairs
        row_pair = True
        col_pair = True
        for cell in pair:
            if not cell.row == row:
                row_pair = False
            if not cell.col == col:
                col_pair = False

        # Check if they are in the same square:
        square_pair = in_same_square(pair)
        
        # Save the row/col pairs and their possible nums
        if row_pair:
            row_pairs.append(tuple_)
        elif col_pair:
            col_pairs.append(tuple_)
        if square_pair:
            square_pairs.append(tuple_)
    
    # The col and row pair have to be in the same square
    tmp_list = []
    for tuple_ in col_pairs:
        (possible_num, pair) = tuple_
        if in_same_square(pair):
            tmp_list.append(tuple_)

    col_pairs = tmp_list

    tmp_list = []
    for tuple_ in row_pairs:
        (possible_num, pair) = tuple_
        if in_same_square(pair):
            tmp_list.append(tuple_)

    row_pairs = tmp_list

    # The square pairs have to be in the same col or row:
    tmp_list = []
    for tuple_ in square_pairs:
        (possible_num, pair) = tuple_
        if all([cell.row == pair[0].row for cell in pair]):
            tmp_list.append(tuple_)
        elif all([cell.col == pair[0].col for cell in pair]):
            tmp_list.append(tuple_)

    square_pairs = tmp_list


    # If it's a row or col pair then all the other cells in the same sqquare is blocked from having the same possible num
    def get_square_index(cell):
        row_i = [all([cell.row < i*3 and cell.row >= (i-1)*3]) for i in range(1,4)]
        row_i = np.argwhere(np.array(row_i) == True)[0,0]
        col_i = [all([cell.col < i*3 and cell.col >= (i-1)*3]) for i in range(1,4)]
        col_i = np.argwhere(np.array(col_i) == True)[0,0]

        return (row_i, col_i)

    changed_something = False
    if cell_type == 'row':
        for tuple_ in row_pairs:
            (possible_num, pair) = tuple_
            # Get the square they are located in
            square_row, square_col = get_square_index(pair[0])
            square = board[3*square_row:3*square_row+3, 3*square_col:3*square_col+3]
            
            for cell in square.reshape(-1):
                if not cell in pair:
                    changed_something |= cell.update_possible_num(possible_num)
            
    
    
    if cell_type == 'col':
        for tuple_ in col_pairs:
            (possible_num, pair) = tuple_
            # Get the square they are located in
            square_row, square_col = get_square_index(pair[0])
            square = board[3*square_row:3*square_row+3, 3*square_col:3*square_col+3]
            
            for cell in square.reshape(-1):
                if not cell in pair:
                    changed_something |= cell.update_possible_num(possible_num)


    # If it's a square pair then all the other cells in the same row or col is blocked from having the same possible num
    if cell_type == 'square':
        for tuple_ in col_pairs:
            (possible_num, pair) = tuple_
            
            cells_to_update = []
            # Get the col/row they are located in
            if pair[0].row == pair[1].row:
                cells_to_update = board[pair[0].row, :]

                # cells_to_update = board[:, pair[0].col]
            elif pair[0].col == pair[1].col:
                cells_to_update = board[:, pair[0].col]

                # cells_to_update = board[pair[0].row, :]

            
            for cell in cells_to_update:
                if not cell in pair:
                    changed_something |= cell.update_possible_num(possible_num)

    return changed_something

def pointing_numbers_row(board, row):
    return pointing_numbers_cells(board, row, 'row')

def pointing_numbers_col(board, col):
    return pointing_numbers_cells(board, col, 'col')

def pointing_numbers_square(board, square):
    return pointing_numbers_cells(board, square, 'square')

# Check if any duplicates in cells
def check_correctness_cells(cells):
    cell_num = get_num_from_cells(cells)
    cell_num = list(cell_num)

    # Remove 0 from cell_num
    while 0 in cell_num:
        cell_num.remove(0)

    # Check for duplicates
    return len(cell_num) == len(set(cell_num)) 

def check_correctness_square(square):
    return check_correctness_cells(square.reshape(-1))
    
def check_correctness_row(row):
    return check_correctness_cells(row)
    
def check_correctness_col(col):
    return check_correctness_cells(col)

def check_correctness_of_board(board):
    # Check the corectness
    correct = True
    for i in range(n_num):
        row = board[i,:]
        correct &= check_correctness_row(row)

    for i in range(n_num):
        col = board[:,i]
        correct &= check_correctness_col(col)

    if n_num == 9:
        for square_i in range(0, n_num):
            row_i = (square_i // 3)*3
            col_i = (square_i % 3)*3
            square = board[row_i:row_i+3, col_i:col_i+3]
            correct &= check_correctness_square(square)
    
    return correct

# Sums up the rows and make sure they all contain numbers 1-9, i.e. the sum is 45
def check_solved(board):
    for row in board:
        if not np.sum(get_num_from_cells(row)) == 45:
            return False

    return True

def solve(board):
    assert board.shape == (9,9), f"The shape of the board must be (9,9). It is {board.shape}."
    # print("Generated board")
    board = cell_board(board)
    # board_print(board)

    # Tries to solve the board using logic. Can't handle any kind of xwing shenanigans
    def try_to_solve(board):
        done = False
        while not done:
            update_something = False

            # Analyze rows
            for i in range(n_num):
                row = board[i,:]
                update_something |= exclusion_cells_row(row)
                update_something |= find_hidden_num_row(row)
                update_something |= pointing_numbers_row(board, row)


            # Analyze cols
            for i in range(n_num):
                col = board[:,i]
                update_something |= exclusion_cells_col(col)
                update_something |= find_hidden_num_col(col)
                update_something |= pointing_numbers_col(board, col)


            # Analyze squares
            if n_num == 9:
                for square_i in range(0, n_num):
                    row_i = (square_i // 3)*3
                    col_i = (square_i % 3)*3
                    square = board[row_i:row_i+3, col_i:col_i+3]
                    update_something |= exclusion_cells_square(square)
                    update_something |= find_hidden_num_square(square)
                    update_something |= pointing_numbers_square(board, square)
                    

            # Update all of the cells
            for cell in board.reshape(-1):
                cell.check_if_only_one_possible_num()

            # If anything was updated then start over if not the soduko is solved
            if not update_something:
                done = True


    # print("Press enter to start solving")
    # input()
    start_time = time.time()
    try_to_solve(board)

    # If it's not solved by now try to randomize a good number
    import copy
    last_known_good_board = copy.deepcopy(board)
    done = check_solved(board)

    for cell in last_known_good_board.reshape(-1):
        for num in cell.possible_nums:
            board[cell.row, cell.col].possible_nums = {num}
            try:
                try_to_solve(board)
            except:
                board = copy.deepcopy(last_known_good_board)

            if check_solved(board):
                done = True
                break
        
        if done:
            break



        
    correct = check_correctness_of_board(board)
    if not correct:
        print("Incorrect solution generated")
        return None
    else:
        print("Done:")



    end_time = time.time()
    print("It took", end_time-start_time, "s to solve!")
    # board_print(board)
    return board


