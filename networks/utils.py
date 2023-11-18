import random


def shuffle_numbers_dict(numbers):
    # Make a copy of the input list to avoid modifying the original list
    shuffled_numbers = numbers.copy()

    # Shuffle the list using random.shuffle()
    random.shuffle(shuffled_numbers)

    # Create a dictionary with original numbers as keys and shuffled numbers as values
    result_dict = {
        original: shuffled for original, shuffled in zip(numbers, shuffled_numbers)
    }

    return result_dict
