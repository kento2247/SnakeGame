class Base:
    def __init__(self):
        self.SCREEN_SIZE = 600
        self.BLOCK_WIDTH = 20
        self.MAX_FOOD_INDEX = (self.SCREEN_SIZE-self.BLOCK_WIDTH)//self.BLOCK_WIDTH
