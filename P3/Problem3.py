def count_islands(grid):
    """
    Counts the number of islands in a 2D grid of 1s (land) and 0s (water).
    An island is formed by connecting adjacent lands horizontally or vertically.
    """

    def marking_visited(row, col):
        """
        Performs a depth-first search (DFS) on the grid starting from the given cell.
        """
        if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] != 1:
            return
        
        grid[row][col] = 2 # marked as visited
        marking_visited(row-1, col)
        marking_visited(row+1, col)
        marking_visited(row, col-1)
        marking_visited(row, col+1)

    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    num_of_island = 0

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1:
                num_of_island += 1
                marking_visited(row, col)

    return num_of_island


map_2d = [[1, 1, 0, 0, 0],[1, 1, 0, 0, 0],[0, 0, 1, 0, 0],[0, 0, 0, 1, 1]]

num_islands = count_islands(map_2d)
print(num_islands)
